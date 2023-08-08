"""
Graph-relational Domain Adaptation (GrDA) for Zero-Shot Cross-Lingual Transfer.

Implements the adversarial training framework from:
    "Graph-relational Domain Adaptation" (Zhao et al., ICLR 2022)

Applied to multi-lingual transfer (ZSCL-R setting):
  - G (LanguageGraphEmbedding): maps one-hot language indices to latent graph embeddings z
  - D (LanguageDiscriminator): predicts edge labels from encoded representations
  - E (LanguageAdaptiveEncoder): language-conditioned feature encoder
  - F (TaskPredictor): task prediction head

Training uses alternating optimization:
    1. Optimize G to reconstruct language graph structure from z embeddings
    2. Optimize D to discriminate edge labels from representations
    3. Optimize E+F adversarially: maximize task prediction while minimizing D's ability to discriminate
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class LanguageGraphEmbedding(nn.Module):
    """
    G network: maps one-hot language indices to latent graph embeddings z.

    Trained to reconstruct the adjacency structure of the language-relational graph.
    The edge reconstruction loss encourages connected languages to have similar z,
    while disconnected languages have dissimilar z.
    """

    def __init__(self, num_languages: int, z_dim: int, hidden_dim: int):
        super().__init__()
        self.num_languages = num_languages
        self.z_dim = z_dim
        self.embed = nn.Linear(num_languages, hidden_dim)
        self.fc = nn.Linear(hidden_dim, z_dim)
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, t_seq: torch.Tensor) -> torch.Tensor:
        """
        t_seq: (num_languages, batch_size, num_languages) — one-hot domain indices
        Returns: z_seq (num_languages * batch_size, z_dim)
        """
        T, B, D = t_seq.shape
        t_flat = t_seq.reshape(T * B, D).float()
        h = F.relu(self.embed(t_flat))
        return self.fc(h)


class LanguageDiscriminator(nn.Module):
    """
    D network: domain discriminator that predicts graph edge labels.

    Given a pair of domain representations (d_i, d_j), predicts whether
    languages i and j are connected in the language-relational graph.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, e_seq: torch.Tensor) -> torch.Tensor:
        """
        e_seq: (num_languages, batch_size, input_dim)
        Returns: d_seq (num_languages, batch_size, input_dim) — domain representations
        """
        T, B, D = e_seq.shape
        e_flat = e_seq.reshape(T * B, D)
        h = F.relu(self.fc1(e_flat))
        d = self.fc2(h)
        return d.reshape(T, B, -1)


class LanguageAdaptiveEncoder(nn.Module):
    """
    E network: language-adaptive encoder conditioned on graph embedding z.

    Combines sentence representation x with language graph embedding z to produce
    a domain-aware feature representation used for both task prediction and adversarial training.
    """

    def __init__(self, input_dim: int, z_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_seq: torch.Tensor, z_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: (num_languages, batch_size, input_dim)
        z_seq: (num_languages * batch_size, z_dim)
        Returns: e_seq (num_languages, batch_size, output_dim)
        """
        T, B, D = x_seq.shape
        x_flat = x_seq.reshape(T * B, D)
        xz = torch.cat([x_flat, z_seq], dim=-1)
        h = F.relu(self.fc1(xz))
        e = self.fc2(h)
        return e.reshape(T, B, -1)


class GrDAModule:
    """
    Full Graph-relational Domain Adaptation training module for ZSCL-R.

    Manages 4 networks (E, F, G, D) with separate optimizers and
    implements the alternating adversarial training procedure.
    """

    def __init__(
        self,
        num_languages: int,
        input_dim: int,
        z_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_pred_classes: int,
        adjacency_matrix: np.ndarray,
        source_language_indices: List[int],
        lambda_gan: float = 0.1,
        sample_v: int = 4,
        sample_v_g: int = 4,
        lr_e: float = 1e-4,
        lr_d: float = 1e-4,
        lr_g: float = 1e-4,
        device: str = "cpu",
    ):
        self.num_languages = num_languages
        self.lambda_gan = lambda_gan
        self.sample_v = min(sample_v, num_languages)
        self.sample_v_g = min(sample_v_g, num_languages)
        self.device = torch.device(device)

        self.A = adjacency_matrix
        self.source_mask = torch.zeros(num_languages, device=self.device)
        for idx in source_language_indices:
            self.source_mask[idx] = 1.0

        self.netG = LanguageGraphEmbedding(num_languages, z_dim, hidden_dim).to(self.device)
        self.netD = LanguageDiscriminator(output_dim, hidden_dim).to(self.device)
        self.netE = LanguageAdaptiveEncoder(
            input_dim, z_dim, hidden_dim, output_dim
        ).to(self.device)
        self.netF = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_pred_classes),
        ).to(self.device)

        self.optimizer_EF = optim.Adam(
            list(self.netE.parameters()) + list(self.netF.parameters()), lr=lr_e
        )
        self.optimizer_D = optim.Adam(self.netD.parameters(), lr=lr_d)
        self.optimizer_G = optim.Adam(
            list(self.netG.parameters()) + [self.netG.weight, self.netG.bias], lr=lr_g
        )

    def _make_t_seq(self, batch_size: int) -> torch.Tensor:
        """Create one-hot domain index tensors for all languages."""
        t_seq = torch.zeros(
            self.num_languages, batch_size, self.num_languages, device=self.device
        )
        for i in range(self.num_languages):
            t_seq[i, :, i] = 1.0
        return t_seq

    def _sample_subgraph(self, n: int) -> List[int]:
        """Sample n random language nodes for subgraph optimization."""
        return np.random.choice(self.num_languages, size=n, replace=False).tolist()

    def _graph_reconstruction_loss(
        self, z_seq: torch.Tensor, subgraph: List[int]
    ) -> torch.Tensor:
        """G loss: reconstruct edge labels of the language graph from z embeddings."""
        criterion = nn.BCEWithLogitsLoss()
        loss = torch.zeros(1, device=self.device)
        count = 0
        for i in range(len(subgraph)):
            for j in range(i + 1, len(subgraph)):
                vi, vj = subgraph[i], subgraph[j]
                label = torch.tensor(float(self.A[vi][vj]), device=self.device)
                stride = z_seq.shape[0] // self.num_languages
                zi = z_seq[vi * stride]
                zj = z_seq[vj * stride]
                output = self.netG.weight * (zi * zj).sum() + self.netG.bias
                loss = loss + criterion(output, label)
                count += 1
        return loss / max(count, 1)

    def _discrimination_loss(
        self, d_seq: torch.Tensor, subgraph: List[int]
    ) -> torch.Tensor:
        """D loss: predict graph edge labels from domain representations."""
        criterion = nn.BCEWithLogitsLoss()
        B = d_seq.shape[1]
        loss_conn = torch.zeros(1, device=self.device)
        loss_disc = torch.zeros(1, device=self.device)
        cnt_c, cnt_d = 0, 0

        for i in range(len(subgraph)):
            for j in range(i + 1, len(subgraph)):
                vi, vj = subgraph[i], subgraph[j]
                label = torch.full((B,), float(self.A[vi][vj]), device=self.device)
                output = (d_seq[vi] * d_seq[vj]).sum(1)
                if self.A[vi][vj]:
                    loss_conn = loss_conn + criterion(output, label)
                    cnt_c += 1
                else:
                    loss_disc = loss_disc + criterion(output, label)
                    cnt_d += 1

        if cnt_c == 0 or cnt_d == 0:
            total = cnt_c + cnt_d
            return (loss_conn + loss_disc) / max(total, 1)
        return 0.5 * (loss_conn / cnt_c + loss_disc / cnt_d) * self.num_languages

    def train_step(
        self,
        x_seq: torch.Tensor,
        y_seq: torch.Tensor,
    ) -> Dict[str, float]:
        """
        One GrDA training step.

        x_seq: (num_languages, batch_size, input_dim)  — encoder CLS representations
        y_seq: (num_languages, batch_size)              — labels (-1 for unlabeled)
        """
        B = x_seq.shape[1]
        t_seq = self._make_t_seq(B)

        z_seq = self.netG(t_seq)
        e_seq = self.netE(x_seq, z_seq)
        d_seq = self.netD(e_seq)

        sg_g = self._sample_subgraph(self.sample_v_g)
        sg_d = self._sample_subgraph(self.sample_v)

        for net in [self.netG, self.netD, self.netE, self.netF]:
            net.train()

        # Step 1: optimize G
        self.netD.eval()
        self.netE.eval()
        self.netF.eval()
        self.optimizer_G.zero_grad()
        loss_G = self._graph_reconstruction_loss(z_seq, sg_g)
        loss_G.backward(retain_graph=True)
        self.optimizer_G.step()

        # Step 2: optimize D
        self.netD.train()
        self.netG.eval()
        self.optimizer_D.zero_grad()
        loss_D = self._discrimination_loss(d_seq, sg_d)
        loss_D.backward(retain_graph=True)
        self.optimizer_D.step()

        # Step 3: optimize E+F adversarially
        self.netE.train()
        self.netF.train()
        self.netD.eval()
        self.optimizer_EF.zero_grad()

        source_idx = (self.source_mask == 1).nonzero(as_tuple=True)[0]
        n_src = len(source_idx)
        if n_src > 0:
            src_e = e_seq[source_idx].reshape(n_src * B, -1)
            src_y = y_seq[source_idx].reshape(n_src * B)
            f_out = self.netF(src_e)
            loss_pred = F.cross_entropy(f_out, src_y, ignore_index=-1)
        else:
            loss_pred = torch.tensor(0.0, device=self.device)

        loss_D_adv = self._discrimination_loss(d_seq, sg_d)
        loss_EF = loss_pred - self.lambda_gan * loss_D_adv
        loss_EF.backward()
        self.optimizer_EF.step()

        return {
            "loss_G": loss_G.item(),
            "loss_D": loss_D.item(),
            "loss_pred": loss_pred.item(),
            "loss_EF": loss_EF.item(),
        }

    def get_adapted_repr(self, x: torch.Tensor, lang_idx: int, batch_size: int) -> torch.Tensor:
        """Get adapted representation for inference (lang_idx = target language index)."""
        with torch.no_grad():
            t_single = torch.zeros(1, batch_size, self.num_languages, device=self.device)
            t_single[0, :, lang_idx] = 1.0
            z = self.netG(t_single)
            x_expanded = x.unsqueeze(0)
            e = self.netE(x_expanded, z)
            return e.squeeze(0)
