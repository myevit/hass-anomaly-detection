import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, D]
        x = self.value_embedding(x)
        return self.dropout(x)


class AutoCorrelation(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        batch, head, channel, length = values.shape
        top_k = int(self.factor * np.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(mean_value, top_k, dim=-1)[1]
        weights = torch.topk(mean_value, top_k, dim=-1)[0]

        # Reshape to [batch, top_k, length]
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values.permute(0, 2, 3, 1).reshape(batch, channel, length, head)

        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[0, i]), dims=-2)
            pattern = pattern.permute(0, 3, 1, 2)
            delays_agg = delays_agg + pattern * (
                tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            )
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        batch, head, channel, length = values.shape

        # index init
        init_index = (
            torch.arange(length)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch, head, channel, 1)
            .to(values.device)
        )

        # find top k
        top_k = int(self.factor * np.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[0]
        delay = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]

        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)

        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(
                1
            ).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (
                tmp_corr[:, i]
                .unsqueeze(1)
                .unsqueeze(1)
                .unsqueeze(1)
                .repeat(1, head, channel, length)
            )
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, : (L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # Period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        # Time delay agg
        if self.training:
            V = self.time_delay_agg_training(
                values.permute(0, 2, 3, 1).contiguous(), corr
            ).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(
                values.permute(0, 2, 3, 1).contiguous(), corr
            ).permute(0, 3, 1, 2)

        if self.output_attention:
            return V.contiguous(), corr.permute(0, 3, 1, 2)
        else:
            return V.contiguous(), None


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads):
        super(AutoCorrelationLayer, self).__init__()
        self.correlation = correlation
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Projection
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # Auto-correlation mechanism
        out, attn = self.correlation(queries, keys, values, attn_mask)

        # Final projection
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        # Autocorrelation module
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        # FFN module
        y = x.transpose(1, 2)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        y = y.transpose(1, 2)

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class SeriesDecomp(nn.Module):
    """Series decomposition block from Autoformer"""

    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = torch.nn.AvgPool1d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )

    def forward(self, x):
        # x: [Batch, Length, Channel]
        moving_mean = self.moving_avg(x.transpose(1, 2)).transpose(1, 2)
        residual = x - moving_mean
        return moving_mean, residual


class DecoderLayer(nn.Module):
    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        c_out,
        d_ff=None,
        dropout=0.1,
        activation="relu",
        moving_avg=25,
    ):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.decomp3 = SeriesDecomp(moving_avg)
        self.projection = nn.Conv1d(
            in_channels=d_model,
            out_channels=c_out,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
        )

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # Self attention
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])

        # Series decomposition
        mean1, res1 = self.decomp1(x)

        # Cross attention
        mean1 = mean1 + self.dropout(
            self.cross_attention(mean1, cross, cross, attn_mask=cross_mask)[0]
        )

        # Series decomposition
        mean2, res2 = self.decomp2(mean1)

        # FFN
        mean2 = mean2.transpose(1, 2)
        mean2 = self.dropout(self.activation(self.conv1(mean2)))
        mean2 = self.dropout(self.conv2(mean2))
        mean2 = mean2.transpose(1, 2)

        # Series decomposition
        mean3, res3 = self.decomp3(mean2)

        # Final projection
        seasonal_part = res1 + res2 + res3
        seasonal_part = seasonal_part.transpose(1, 2)
        seasonal_part = self.projection(seasonal_part)
        seasonal_part = seasonal_part.transpose(1, 2)

        return seasonal_part, mean3


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        return x, trend


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len, target_dim=None):
        """
        Initialize time series dataset for Autoformer

        Args:
            data: numpy array of shape [seq_length, n_features]
            seq_len: input sequence length
            pred_len: prediction sequence length
            target_dim: target dimension for prediction (None means all dimensions)
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_dim = target_dim

        # Determine total samples
        self.samples = len(data) - seq_len - pred_len + 1

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]

        if self.target_dim is not None:
            seq_y = seq_y[:, self.target_dim]

        return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y)


class AnomalyDetector:
    def __init__(
        self, model_path=None, seq_len=100, pred_len=25, enc_in=4, dec_in=4, c_out=4
    ):
        """
        Initialize Autoformer Anomaly Detector

        Args:
            model_path: path to load pretrained model from
            seq_len: input sequence length
            pred_len: prediction sequence length
            enc_in: number of input features
            dec_in: number of decoder input features
            c_out: number of output features
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out

        # Model parameters
        d_model = 64
        n_heads = 4
        e_layers = 2
        d_layers = 1
        d_ff = 256
        dropout = 0.1
        factor = 3  # Auto-correlation factor
        moving_avg = 25  # Moving average window for decomposition

        # Initialize model components
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            True,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    c_out,
                    d_ff,
                    dropout,
                    moving_avg=moving_avg,
                )
                for _ in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True),
        )

        # Decomposition
        self.decomp = SeriesDecomp(moving_avg)

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to_device()

        # Load pretrained model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

        # Initialize error statistics
        self.mean_error = 0
        self.std_error = 1
        self.is_calibrated = False

    def to_device(self):
        """Move all model components to the device"""
        self.enc_embedding = self.enc_embedding.to(self.device)
        self.dec_embedding = self.dec_embedding.to(self.device)
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.decomp = self.decomp.to(self.device)

    def train(self, train_data, epochs=10, batch_size=32, learning_rate=0.0001):
        """
        Train the Autoformer model

        Args:
            train_data: numpy array of shape [seq_length, n_features]
            epochs: number of training epochs
            batch_size: batch size
            learning_rate: learning rate
        """
        # Set training mode
        self.enc_embedding.train()
        self.dec_embedding.train()
        self.encoder.train()
        self.decoder.train()
        self.decomp.train()

        # Create optimizer
        parameters = (
            list(self.enc_embedding.parameters())
            + list(self.dec_embedding.parameters())
            + list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.decomp.parameters())
        )

        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        criterion = nn.MSELoss()

        # Create dataset and dataloader
        dataset = TimeSeriesDataset(train_data, self.seq_len, self.pred_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Prepare decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.pred_len :, :]).to(
                    self.device
                )
                dec_inp = torch.cat(
                    [batch_x[:, -self.seq_len // 4 :, :], dec_inp], dim=1
                )

                # Forward pass
                outputs = self.forward(batch_x, dec_inp)

                # Compute loss
                loss = criterion(outputs, batch_y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

        # Calibrate error statistics after training
        self.calibrate_error_stats(train_data)

    def forward(self, x_enc, x_dec):
        """
        Forward pass through the Autoformer model

        Args:
            x_enc: encoder input [batch_size, seq_len, enc_in]
            x_dec: decoder input [batch_size, pred_len, dec_in]

        Returns:
            predictions [batch_size, pred_len, c_out]
        """
        # Decompose encoder input
        mean_x, _ = self.decomp(x_enc)

        # Embedding
        enc_out = self.enc_embedding(x_enc)
        dec_out = self.dec_embedding(x_dec)

        # Encoder
        enc_out, _ = self.encoder(enc_out)

        # Decoder
        dec_out, trend = self.decoder(dec_out, enc_out, trend=mean_x)

        # Final output
        dec_out = dec_out + trend[:, -self.pred_len :, :]

        return dec_out

    def predict(self, data):
        """
        Make predictions using the trained model

        Args:
            data: numpy array of shape [seq_length, n_features]

        Returns:
            predictions: numpy array of shape [pred_length, n_features]
        """
        # Set evaluation mode
        self.enc_embedding.eval()
        self.dec_embedding.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.decomp.eval()

        with torch.no_grad():
            # Format input
            x_enc = (
                torch.FloatTensor(data[-self.seq_len :]).unsqueeze(0).to(self.device)
            )

            # Prepare decoder input (zeros for future prediction)
            x_dec = torch.zeros((1, self.pred_len, self.dec_in)).float().to(self.device)
            x_dec = torch.cat([x_enc[:, -self.seq_len // 4 :, :], x_dec], dim=1)

            # Forward pass
            predictions = self.forward(x_enc, x_dec)

            return predictions.cpu().numpy()[0]

    def calibrate_error_stats(self, data):
        """
        Calibrate error statistics on training data for anomaly thresholds

        Args:
            data: numpy array of training data [seq_length, n_features]
        """
        all_errors = []

        # Create dataset without shuffling to maintain sequence order
        dataset = TimeSeriesDataset(data, self.seq_len, self.pred_len)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        # Calculate reconstruction error for all sequences
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)

                # Prepare decoder input
                dec_inp = (
                    torch.zeros((batch_x.shape[0], self.pred_len, self.dec_in))
                    .float()
                    .to(self.device)
                )
                dec_inp = torch.cat(
                    [batch_x[:, -self.seq_len // 4 :, :], dec_inp], dim=1
                )

                # Forward pass
                predictions = self.forward(batch_x, dec_inp)

                # Calculate MSE error
                mse = torch.mean(
                    (predictions - batch_y.to(self.device)) ** 2, dim=(1, 2)
                )
                all_errors.append(mse.cpu().numpy())

        # Concatenate all errors
        all_errors = np.concatenate(all_errors)

        # Calculate mean and std
        self.mean_error = np.mean(all_errors)
        self.std_error = np.std(all_errors)
        self.is_calibrated = True

        print(
            f"Calibrated error statistics - Mean: {self.mean_error:.4f}, Std: {self.std_error:.4f}"
        )

    def detect_anomaly(self, data, threshold_sigmas=3.0):
        """
        Detect if the given data contains anomalies

        Args:
            data: numpy array of shape [seq_length, n_features]
            threshold_sigmas: number of standard deviations for anomaly threshold

        Returns:
            is_anomaly: boolean indicating if an anomaly was detected
            anomaly_score: the anomaly score (higher means more anomalous)
            normalized_score: score normalized to 0-1 range using sigmoid
        """
        if not self.is_calibrated:
            print("Warning: Model not calibrated. Using default error statistics.")

        # Ensure we have enough data
        if len(data) < self.seq_len:
            print(
                f"Warning: Not enough data for prediction. Need {self.seq_len}, got {len(data)}"
            )
            # Pad with zeros if needed
            pad_length = self.seq_len - len(data)
            data = np.vstack([np.zeros((pad_length, data.shape[1])), data])

        # Get the input sequence
        input_seq = data[-self.seq_len :]

        # Make prediction
        predictions = self.predict(data)

        # Calculate prediction error (MSE)
        target_seq = (
            data[-self.pred_len :]
            if len(data) >= self.seq_len + self.pred_len
            else data[-self.pred_len :]
        )
        prediction_error = np.mean((predictions - target_seq) ** 2)

        # Calculate anomaly threshold
        anomaly_threshold = self.mean_error + threshold_sigmas * self.std_error

        # Determine if this is an anomaly
        is_anomaly = prediction_error > anomaly_threshold

        # Calculate normalized score between 0 and 1 using sigmoid
        z_score = (prediction_error - self.mean_error) / (self.std_error + 1e-10)
        normalized_score = 1 / (1 + np.exp(-z_score + threshold_sigmas / 2))

        return is_anomaly, prediction_error, normalized_score

    def save_model(self, path):
        """Save the model to the specified path"""
        torch.save(
            {
                "enc_embedding": self.enc_embedding.state_dict(),
                "dec_embedding": self.dec_embedding.state_dict(),
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "decomp": self.decomp.state_dict(),
                "mean_error": self.mean_error,
                "std_error": self.std_error,
                "is_calibrated": self.is_calibrated,
                "config": {
                    "seq_len": self.seq_len,
                    "pred_len": self.pred_len,
                    "enc_in": self.enc_in,
                    "dec_in": self.dec_in,
                    "c_out": self.c_out,
                },
            },
            path,
        )
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load the model from the specified path"""
        checkpoint = torch.load(path, map_location=self.device)

        # Load configuration if available
        if "config" in checkpoint:
            config = checkpoint["config"]
            self.seq_len = config["seq_len"]
            self.pred_len = config["pred_len"]
            self.enc_in = config["enc_in"]
            self.dec_in = config["dec_in"]
            self.c_out = config["c_out"]

        # Load model components
        self.enc_embedding.load_state_dict(checkpoint["enc_embedding"])
        self.dec_embedding.load_state_dict(checkpoint["dec_embedding"])
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.decomp.load_state_dict(checkpoint["decomp"])

        # Load error statistics if available
        if "mean_error" in checkpoint:
            self.mean_error = checkpoint["mean_error"]
        if "std_error" in checkpoint:
            self.std_error = checkpoint["std_error"]
        if "is_calibrated" in checkpoint:
            self.is_calibrated = checkpoint["is_calibrated"]

        print(f"Model loaded from {path}")
