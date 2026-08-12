import torch
import torch.nn as nn
# from torch.nn import TransformerEncoder
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
# import wandb
from datetime import datetime

from ticl.models.encoders import OneHotAndLinear
from ticl.models.decoders import MLPModelDecoder
from ticl.models.layer import TransformerEncoderLayer, TransformerEncoderSimple
from ticl.models.encoders import Linear

from ticl.utils import SeqBN, get_init_method

from sklearn.preprocessing import QuantileTransformer

class MotherNet(torch.nn.Module):
    # n_out is 1 in regression
    def __init__(self, *, n_out, emsize, nhead, nhid_factor, nlayers, n_features, dropout=0.0, y_encoder_layer=None,
                 input_normalization=False, init_method=None, pre_norm=False,
                 activation='gelu', recompute_attn=False,
                 all_layers_same_init=False, efficient_eval_masking=True, decoder_type="output_attention", predicted_hidden_layer_size=None,
                 cls_token_n=0, decoder_embed_dim=2048, classification_task=True,
                 decoder_hidden_layers=1, decoder_hidden_size=None, predicted_hidden_layers=1, weight_embedding_rank=None, y_encoder=None,
                 low_rank_weights=False, tabpfn_zero_weights=True, decoder_activation="relu", predicted_activation="relu",
                 bins_output_n=0, bins_to_sum_n=0, low_rank_last=False):
        super().__init__()
        self.classification_task = classification_task
        # decoder activation = "relu" is legacy behavior
        nhid = emsize * nhid_factor
        # mothernet has batch_first=False, unlike all the other models.
        def encoder_layer_creator(): return TransformerEncoderLayer(emsize, nhead, nhid, dropout, activation=activation,
                                                                    pre_norm=pre_norm, recompute_attn=recompute_attn, batch_first=False)
        self.transformer_encoder = TransformerEncoderSimple(encoder_layer_creator, nlayers)
        
        backbone_size = sum(p.numel() for p in self.transformer_encoder.parameters())
        # if wandb.run: wandb.log({"backbone_size": backbone_size})
        print("Number of parameters in backbone: ", backbone_size)

        self.decoder_activation = decoder_activation
        self.emsize = emsize
        self.encoder = Linear(n_features, emsize, replace_nan_by_zero=True)
        self.c_emb = torch.nn.Embedding(2, emsize)
        self.y_encoder = y_encoder_layer
        self.input_ln = SeqBN(emsize) if input_normalization else None
        self.init_method = init_method
        self.efficient_eval_masking = efficient_eval_masking
        self.n_out = bins_output_n if bins_output_n > 0 else 1
        self.nhid = nhid
        self.decoder_type = decoder_type
        decoder_hidden_size = decoder_hidden_size or nhid
        self.tabpfn_zero_weights = tabpfn_zero_weights
        self.predicted_activation = predicted_activation
        self.bins_output_n = bins_output_n
        self.low_rank_last = low_rank_last

        self.decoder = MLPModelDecoder(emsize=emsize, hidden_size=decoder_hidden_size, n_out=self.n_out, decoder_type=self.decoder_type,
                                       predicted_hidden_layer_size=predicted_hidden_layer_size, embed_dim=decoder_embed_dim,
                                       decoder_hidden_layers=decoder_hidden_layers, nhead=nhead, predicted_hidden_layers=predicted_hidden_layers,
                                       weight_embedding_rank=weight_embedding_rank, low_rank_weights=low_rank_weights, decoder_activation=decoder_activation,
                                       in_size=n_features, cls_token_n=cls_token_n, low_rank_last=low_rank_last)
        if decoder_type in ["special_token", "cls_token"]:
            self.token_embedding = nn.Parameter(torch.randn(cls_token_n, 1, emsize))

        self.init_weights()

    def init_weights(self):
        if self.init_method is not None:
            self.apply(get_init_method(self.init_method))
        if self.tabpfn_zero_weights:
            for layer in self.transformer_encoder.layers:
                nn.init.zeros_(layer.linear2.weight)
                nn.init.zeros_(layer.linear2.bias)
                attns = layer.self_attn if isinstance(layer.self_attn, nn.ModuleList) else [layer.self_attn]
                for attn in attns:
                    nn.init.zeros_(attn.out_proj.weight)
                    nn.init.zeros_(attn.out_proj.bias)

    def inner_forward(self, train_x):
        return self.transformer_encoder(train_x)
    
    def calc_nn_weights(self, x_train: torch.Tensor, 
                        y_train: torch.Tensor, 
                        c_train: torch.Tensor):
        x_enc = self.encoder(x_train)
        y_enc = self.y_encoder(y_train[..., None])
        c_enc = self.c_emb(c_train.ravel()).reshape(y_enc.shape)
        enc_train = x_enc + y_enc + c_enc
        
        if self.decoder_type in ["cls_token", "class_tokens"]:
            cls_tokens_rep = self.token_embedding.repeat(1, enc_train.shape[1], 1)
            enc_train = torch.cat((cls_tokens_rep, enc_train), dim=0)

        output = self.inner_forward(enc_train)
        return self.decoder(output, y_train)
    
    def nn_forward(self, x_test: torch.Tensor, layers):
        (b1, w1), *layers = layers
        x_test_nona = torch.nan_to_num(x_test, nan=0)
        h = (x_test_nona.unsqueeze(-1) * w1.unsqueeze(0)).sum(2)

        if self.decoder.weight_embedding_rank is not None and len(layers):
            h = torch.matmul(h, self.decoder.shared_weights[0])
        h = h + b1

        for i, (b, w) in enumerate(layers):
            if self.predicted_activation == "relu":
                h = torch.relu(h)
            elif self.predicted_activation == "gelu":
                h = torch.nn.functional.gelu(h)
            else:
                raise ValueError(f"Unsupported predicted activation: {self.predicted_activation}")
            if i < len(layers) - 1:
                h = (h.unsqueeze(-1) * w.unsqueeze(0)).sum(2)
                if self.decoder.weight_embedding_rank is not None: 
                    # last layer has no shared weights
                    h = torch.matmul(h, self.decoder.shared_weights[i + 1])
            else:
                if self.low_rank_last:
                    h = torch.matmul(h, self.decoder.shared_weights[i + 1])
                    h = (h.unsqueeze(-1) * w.unsqueeze(0)).sum(2)
                else:
                    h = (h.unsqueeze(-1) * w.unsqueeze(0)).sum(2)
            h = h + b

        # if h.isnan().all():
        #     print("NAN")
        #     raise ValueError("NAN")
        return h
    
    def forward(self, src, single_eval_pos=None):
        assert isinstance(src, tuple), 'inputs (src) have to be given as (x, y, c)'

        x, y, c = src
        # y_mean = torch.mean(y, dim=0, keepdim=True)
        # y_std = torch.std(y, dim=0, keepdim=True)
        # y_std[y_std < 1e-6] = 1
        
        y_train = y[:single_eval_pos]
        
        layers = self.calc_nn_weights(x[:single_eval_pos],
                                      y_train,
                                      c[:single_eval_pos])
        cate_estim = self.nn_forward(x[single_eval_pos:], layers)
        if torch.all(torch.isnan(cate_estim)):
            with open('nan_log.txt', 'a') as file:
                dt = datetime.today()
                file.write(f'-------[{dt.strftime("%H_%M_%S|%d_%m_%Y")}]-------\n')
                nan_tf_count = 0
                for weights in self.transformer_encoder.parameters():
                    nan_tf_count += torch.count_nonzero(torch.isnan(weights))
                msg = 'NaN in TF: ' + str(nan_tf_count)
                file.write(msg)
                msg = 'NaN after TF: ' + 'YES' if torch.any(torch.isnan(layers[0][1])) else 'NO' + '\n'
                file.write(msg)
                msg = f'y values (pivot {single_eval_pos}): {y}\n\nmean train: {y[:single_eval_pos].mean(0)}, std train: {y[:single_eval_pos].std(0)}\n'
                file.write(msg)
        if self.bins_output_n > 0:
            y_min = y_train.amin(dim=0, keepdim=True)
            y_max = y_train.amax(dim=0, keepdim=True)
            ruler_max = y_max - y_min
            ruler_min = y_min - y_max
            ruler_diff = (ruler_max - ruler_min) / self.bins_output_n
            idx = torch.arange(self.bins_output_n, device=cate_estim.device)[None, None, :].repeat(cate_estim.shape[0], 1, 1)
            y_ruler = idx * ruler_diff[..., None] + ruler_min[..., None]
            cate_estim = (cate_estim, y_ruler)
        return cate_estim
                
        
    def get_device(self):
        return self.decoder.shared_weights[0].device

    def pad_zeros(self, x: np.ndarray):
        feats_n = self.encoder.in_features
        if x.shape[-1] > feats_n:
            raise RuntimeError(f"Maximum number of features is {feats_n}")
        elif x.shape[-1] == feats_n:
            return x
        shape = x.shape[:-1] + (feats_n - x.shape[-1],)
        return np.concatenate((x, np.zeros(shape)), axis=-1)

    def _validate_support(self, X, y, c):
        try:
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            c = np.asarray(c)
        except (TypeError, ValueError) as error:
            raise ValueError("X, y, and treatment must be numeric arrays") from error
        if X.ndim not in (2, 3):
            raise ValueError("X must be a non-empty 2-D or 3-D array")
        if X.shape[0] == 0 or X.shape[-1] == 0:
            raise ValueError("X must be a non-empty 2-D or 3-D array")
        if y.shape != X.shape[:-1] or c.shape != X.shape[:-1]:
            raise ValueError("X, y, and treatment shape must match")
        if X.shape[-1] > self.encoder.in_features:
            raise ValueError(
                f"X must have at most {self.encoder.in_features} features"
            )
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X and y must contain only finite values")
        if not np.isin(c, (0, 1)).all():
            raise ValueError("treatment must be binary with values 0 and 1")
        c = c.astype(np.int64, copy=False)
        treatment_tasks = c[:, None] if c.ndim == 1 else c
        for task in range(treatment_tasks.shape[1]):
            if set(np.unique(treatment_tasks[:, task])) != {0, 1}:
                raise ValueError("treatment must contain both treatment arms per task")
        return X, y, c

    def _fit_quantile_transform(self, X: np.ndarray) -> np.ndarray:
        n_quantiles = max(1, X.shape[0] // 5)
        if X.ndim == 2:
            self.q_tf = QuantileTransformer(
                output_distribution='normal', n_quantiles=n_quantiles
            )
            return self.q_tf.fit_transform(X)
        self.q_tf = []
        transformed = []
        for task in range(X.shape[1]):
            transformer = QuantileTransformer(
                output_distribution='normal', n_quantiles=n_quantiles
            )
            self.q_tf.append(transformer)
            transformed.append(transformer.fit_transform(X[:, task, :]))
        return np.stack(transformed, axis=1)

    def _transform_query(self, X: np.ndarray) -> np.ndarray:
        if self._fit_ndim == 2:
            return self.q_tf.transform(X)
        transformed = [
            transformer.transform(X[:, task, :])
            for task, transformer in enumerate(self.q_tf)
        ]
        return np.stack(transformed, axis=1)
    
    @torch.inference_mode()    
    def fit(self, X: np.ndarray, y: np.ndarray, c: np.ndarray):
        X, y, c = self._validate_support(X, y, c)
        self._fit_ndim = X.ndim
        self.n_features_in_ = X.shape[-1]
        self._n_tasks = 1 if X.ndim == 2 else X.shape[1]
        X = self._fit_quantile_transform(X)
        X = self.pad_zeros(X)
        tr_func = lambda arg: torch.from_numpy(arg[0]).to(device=self.get_device(), dtype=arg[1])
        df_dt = torch.get_default_dtype()
        X_t, y_t, c_t = map(tr_func, ((X, df_dt), (y, df_dt), (c, torch.long)))
        if X.ndim == 2:
            X_t = X_t[:, None, :]
            y_t = y_t[:, None]
            c_t = c_t[:, None]
        mask_0 = c_t == 0
        mask_1 = c_t == 1
        y_mean_0 = torch.where(mask_0, y_t, 0).sum(dim=0, keepdim=True) / mask_0.sum(dim=0, keepdim=True)
        y_mean_1 = torch.where(mask_1, y_t, 0).sum(dim=0, keepdim=True) / mask_1.sum(dim=0, keepdim=True)
        y_std = torch.std(y_t, dim=0, keepdim=True)
        y_std[y_std < 1e-6] = 1
        self.y_std = y_std
        self.y_mean_0 = y_mean_0
        self.y_mean_1 = y_mean_1
        y_t = torch.where(
            mask_0,
            (y_t - y_mean_0) / y_std,
            (y_t - y_mean_1) / y_std,
        )
        # print(f"Fitting model for {X_t.shape[1]} task(s) with {X_t.shape[0]} points")
        self.pred_layers = self.calc_nn_weights(X_t, y_t, c_t)
        if self.bins_output_n > 0:
            y_min = y_t.amin(dim=0, keepdim=True)
            y_max = y_t.amax(dim=0, keepdim=True)
            ruler_max = y_max - y_min
            ruler_min = y_min - y_max
            ruler_diff = (ruler_max - ruler_min) / self.bins_output_n
            idx = torch.arange(self.bins_output_n, device=self.get_device())[None, :]
            self.y_ruler = idx * ruler_diff.T + ruler_min.T
        return self
    
    @torch.inference_mode()
    def predict(self, X: np.ndarray, batch_size: int = 1024):
        layers = getattr(self, 'pred_layers', None)
        if layers is None:
            raise RuntimeError("Fit model first")
        try:
            X = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("X must be a numeric array") from error
        if X.ndim != self._fit_ndim:
            raise ValueError("query X must have the same dimensionality as support X")
        if X.shape[0] == 0:
            raise ValueError("query X must be non-empty")
        if X.shape[-1] != self.n_features_in_:
            raise ValueError(
                f"query X must have the same {self.n_features_in_} features as support X"
            )
        if X.ndim == 3 and X.shape[1] != self._n_tasks:
            raise ValueError(
                f"query X must have the same {self._n_tasks} tasks as support X"
            )
        if not np.isfinite(X).all():
            raise ValueError("query X must contain only finite values")
        X = self._transform_query(X)
        X = self.pad_zeros(X)
        if X.ndim == 2:
            X = X[:, None, :]
        X_t = torch.tensor(X).to(device=self.get_device(), dtype=torch.get_default_dtype())
        ds = TensorDataset(X_t)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        pred_idx_list = []
        for x_b, in dl:
            cur_pred = self.nn_forward(x_b, layers)
            if self.bins_output_n > 0:
                pred_idx_list.append(
                    torch.argmax(cur_pred, dim=-1)
                )
            else:
                pred_idx_list.append(cur_pred.squeeze(-1))
        pred_idx = torch.cat(pred_idx_list, dim=0)
        if self.bins_output_n > 0:
            y = torch.gather(self.y_ruler, 1, pred_idx.T).T
        else:
            y = pred_idx
        y = y * self.y_std + (self.y_mean_1 - self.y_mean_0)
        pred = y.cpu().numpy()
        if self._fit_ndim == 2:
            return pred[:, 0].reshape(-1)
        return pred.reshape(X.shape[0], self._n_tasks)
