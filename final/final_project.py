import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv, GATConv, GCNConv
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from transformers import (
    AutoTokenizer, 
    AutoModel,
    RobertaTokenizer,
    RobertaModel,
    BertTokenizer,
    BertModel,
    DistilBertTokenizer,
    DistilBertModel,
    AlbertTokenizer,
    AlbertModel,
)
from tqdm import tqdm
import pickle
from dataclasses import dataclass
import json


@dataclass
class EmbeddingConfig:
    model_name: str
    max_length: int = 512
    batch_size: int = 32
    pooling_strategy: str = 'cls'  # 'cls', 'mean', 'max'
    use_title_only: bool = False
    use_abstract_only: bool = False
    normalize: bool = True


class TextEmbeddingGenerator:
    def __init__(self, config: EmbeddingConfig, device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # Load tokenizer and model
        print(f"Loading {config.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name).to(device)
        self.model.eval()
        
        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def pool_embeddings(self, hidden_states, attention_mask):
        if self.config.pooling_strategy == 'cls':
            # Use [CLS] token (first token)
            return hidden_states[:, 0, :]
        
        elif self.config.pooling_strategy == 'mean':
            # Mean pooling with attention mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask
        
        elif self.config.pooling_strategy == 'max':
            # Max pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states[input_mask_expanded == 0] = -1e9
            return torch.max(hidden_states, 1)[0]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.config.pooling_strategy}")
    
    @torch.no_grad()
    def generate_embeddings(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        all_embeddings = []
        
        # Process in batches
        num_batches = (len(texts) + self.config.batch_size - 1) // self.config.batch_size
        iterator = range(0, len(texts), self.config.batch_size)
        
        if show_progress:
            iterator = tqdm(iterator, total=num_batches, desc="Generating embeddings")
        
        for i in iterator:
            batch_texts = texts[i:i + self.config.batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors='pt'
            ).to(self.device)
            
            # Get embeddings
            outputs = self.model(**encoded)
            hidden_states = outputs.last_hidden_state
            
            # Pool
            embeddings = self.pool_embeddings(hidden_states, encoded['attention_mask'])
            
            # Normalize if requested
            if self.config.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def save_embeddings(self, embeddings: np.ndarray, save_path: str):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, embeddings)
        print(f"Embeddings saved to {save_path}")
    
    @staticmethod
    def load_embeddings(load_path: str) -> np.ndarray:
        return np.load(load_path)


class TextDataProcessor:
    def __init__(self, text_file_path: str):
        self.text_file_path = text_file_path
        self.df = None
    
    def load_data(self):
        # Try to determine file format
        if self.text_file_path.endswith('.tsv'):
            self.df = pd.read_csv(self.text_file_path, sep='\t')
        else:
            # Try to auto-detect
            self.df = pd.read_csv(self.text_file_path, sep=None, engine='python')
        
        print(f"Loaded {len(self.df)} papers")
        print(f"Columns: {self.df.columns.tolist()}")
        return self.df
    
    def prepare_texts(self, use_title_only: bool = False, use_abstract_only: bool = False) -> List[str]:
        if self.df is None:
            self.load_data()
        
        texts = []
        for _, row in self.df.iterrows():
            if use_title_only:
                text = str(row['title']) if pd.notna(row['title']) else ""
            elif use_abstract_only:
                text = str(row['abstract']) if pd.notna(row['abstract']) else ""
            else:
                # Combine title and abstract
                title = str(row['title']) if pd.notna(row['title']) else ""
                abstract = str(row['abstract']) if pd.notna(row['abstract']) else ""
                
                if title and abstract:
                    text = f"{title}. {abstract}"
                else:
                    text = title or abstract
            
            texts.append(text)
        
        return texts
    
    def get_node_ids(self) -> np.ndarray:
        if self.df is None:
            self.load_data()
        return self.df['node_id'].values


class OGBArxivWithEmbeddingsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        embeddings: np.ndarray,
        root: str = './data',
        batch_size: int = 1024,
        num_neighbors: List[int] = [10, 10],
        num_workers: int = 4,
    ):
        super().__init__()
        self.embeddings = torch.from_numpy(embeddings).float()
        self.root = root
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors
        self.num_workers = num_workers
        
    def setup(self, stage: Optional[str] = None):
        # Load ogbn-arxiv dataset
        self.dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=self.root)
        self.data = self.dataset[0]
        
        # Replace node features with text embeddings
        print(f"Original features shape: {self.data.x.shape}")
        print(f"Text embeddings shape: {self.embeddings.shape}")
        
        # Ensure embeddings match number of nodes
        assert self.embeddings.shape[0] == self.data.num_nodes, \
            f"Embeddings ({self.embeddings.shape[0]}) must match nodes ({self.data.num_nodes})"
        
        self.data.x = self.embeddings
        
        # Get split indices
        split_idx = self.dataset.get_idx_split()
        self.train_idx = split_idx['train']
        self.val_idx = split_idx['valid']
        self.test_idx = split_idx['test']
        
        # Store properties
        self.num_features = self.embeddings.shape[1]
        self.num_classes = self.dataset.num_classes
        
        print(f"Using {self.num_features} dimensional embeddings")
    
    def train_dataloader(self):
        return NeighborLoader(
            self.data,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            input_nodes=self.train_idx,
            num_workers=self.num_workers,
            shuffle=True,
        )
    
    def val_dataloader(self):
        return NeighborLoader(
            self.data,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            input_nodes=self.val_idx,
            num_workers=self.num_workers,
            shuffle=False,
        )
    
    def test_dataloader(self):
        return NeighborLoader(
            self.data,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            input_nodes=self.test_idx,
            num_workers=self.num_workers,
            shuffle=False,
        )


class GNNClassifier(pl.LightningModule):
    def __init__(
        self,
        gnn_type: str,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.dropout = dropout
        self.gnn_type = gnn_type
        
        # Build GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        if gnn_type.lower() == 'sage':
            self.convs.append(SAGEConv(in_channels, hidden_channels))
        elif gnn_type.lower() == 'gat':
            self.convs.append(GATConv(in_channels, hidden_channels, heads=4, concat=True))
            hidden_channels = hidden_channels * 4  # Account for concatenated heads
        elif gnn_type.lower() == 'gcn':
            self.convs.append(GCNConv(in_channels, hidden_channels))
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            if gnn_type.lower() == 'sage':
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            elif gnn_type.lower() == 'gat':
                self.convs.append(GATConv(hidden_channels, hidden_channels // 4, heads=4, concat=True))
            elif gnn_type.lower() == 'gcn':
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        if gnn_type.lower() == 'sage':
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        elif gnn_type.lower() == 'gat':
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=1, concat=False))
        elif gnn_type.lower() == 'gcn':
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Classifier
        self.classifier = nn.Linear(hidden_channels, num_classes)
        
        # Evaluator
        self.evaluator = Evaluator(name='ogbn-arxiv')
        
        # Store predictions
        self.val_preds = []
        self.val_labels = []
        self.test_preds = []
        self.test_labels = []
    
    def forward(self, x, edge_index, batch_size=None):
        # Apply GNN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = bn(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Get only the target nodes (first batch_size nodes)
        if batch_size is not None:
            x = x[:batch_size]
        
        # Final batch norm
        x = self.batch_norms[-1](x)
        
        # Classify
        logits = self.classifier(x)
        return logits
    
    def training_step(self, batch, batch_idx):
        logits = self(batch.x, batch.edge_index, batch.batch_size)
        y = batch.y[:batch.batch_size].squeeze()
        loss = F.cross_entropy(logits, y)
        
        pred = logits.argmax(dim=-1)
        acc = (pred == y).float().mean()
        
        self.log('train_loss', loss, batch_size=batch.batch_size, prog_bar=True)
        self.log('train_acc', acc, batch_size=batch.batch_size, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch.x, batch.edge_index, batch.batch_size)
        y = batch.y[:batch.batch_size].squeeze()
        loss = F.cross_entropy(logits, y)
        
        pred = logits.argmax(dim=-1)
        self.val_preds.append(pred.cpu())
        self.val_labels.append(y.cpu())
        
        self.log('val_loss', loss, batch_size=batch.batch_size)
        return loss
    
    def on_validation_epoch_end(self):
        if len(self.val_preds) > 0:
            y_true = torch.cat(self.val_labels).unsqueeze(-1).numpy()
            y_pred = torch.cat(self.val_preds).unsqueeze(-1).numpy()
            
            result = self.evaluator.eval({'y_true': y_true, 'y_pred': y_pred})
            val_acc = result['acc']
            
            self.log('val_acc', val_acc, prog_bar=True)
            
            self.val_preds.clear()
            self.val_labels.clear()
    
    def test_step(self, batch, batch_idx):
        logits = self(batch.x, batch.edge_index, batch.batch_size)
        y = batch.y[:batch.batch_size].squeeze()
        
        pred = logits.argmax(dim=-1)
        self.test_preds.append(pred.cpu())
        self.test_labels.append(y.cpu())
    
    def on_test_epoch_end(self):
        if len(self.test_preds) > 0:
            y_true = torch.cat(self.test_labels).unsqueeze(-1).numpy()
            y_pred = torch.cat(self.test_preds).unsqueeze(-1).numpy()
            
            result = self.evaluator.eval({'y_true': y_true, 'y_pred': y_pred})
            test_acc = result['acc']
            
            self.log('test_acc', test_acc)
            print(f"\nTest Accuracy: {test_acc:.4f}")
            
            self.test_preds.clear()
            self.test_labels.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_acc',
            }
        }


class MLPClassifier(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        layers = []
        
        # Input layer
        layers.extend([
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        # Output layer
        layers.append(nn.Linear(hidden_channels, num_classes))
        
        self.mlp = nn.Sequential(*layers)
        
        # Evaluator
        self.evaluator = Evaluator(name='ogbn-arxiv')
        
        # Store predictions
        self.val_preds = []
        self.val_labels = []
        self.test_preds = []
        self.test_labels = []
    
    def forward(self, x, edge_index=None, batch_size=None):
        if batch_size is not None:
            x = x[:batch_size]
        return self.mlp(x)
    
    def training_step(self, batch, batch_idx):
        logits = self(batch.x, batch_size=batch.batch_size)
        y = batch.y[:batch.batch_size].squeeze()
        loss = F.cross_entropy(logits, y)
        
        pred = logits.argmax(dim=-1)
        acc = (pred == y).float().mean()
        
        self.log('train_loss', loss, batch_size=batch.batch_size, prog_bar=True)
        self.log('train_acc', acc, batch_size=batch.batch_size, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch.x, batch_size=batch.batch_size)
        y = batch.y[:batch.batch_size].squeeze()
        loss = F.cross_entropy(logits, y)
        
        pred = logits.argmax(dim=-1)
        self.val_preds.append(pred.cpu())
        self.val_labels.append(y.cpu())
        
        self.log('val_loss', loss, batch_size=batch.batch_size)
        return loss
    
    def on_validation_epoch_end(self):
        if len(self.val_preds) > 0:
            y_true = torch.cat(self.val_labels).unsqueeze(-1).numpy()
            y_pred = torch.cat(self.val_preds).unsqueeze(-1).numpy()
            
            result = self.evaluator.eval({'y_true': y_true, 'y_pred': y_pred})
            val_acc = result['acc']
            
            self.log('val_acc', val_acc, prog_bar=True)
            
            self.val_preds.clear()
            self.val_labels.clear()
    
    def test_step(self, batch, batch_idx):
        logits = self(batch.x, batch_size=batch.batch_size)
        y = batch.y[:batch.batch_size].squeeze()
        
        pred = logits.argmax(dim=-1)
        self.test_preds.append(pred.cpu())
        self.test_labels.append(y.cpu())
    
    def on_test_epoch_end(self):
        if len(self.test_preds) > 0:
            y_true = torch.cat(self.test_labels).unsqueeze(-1).numpy()
            y_pred = torch.cat(self.test_preds).unsqueeze(-1).numpy()
            
            result = self.evaluator.eval({'y_true': y_true, 'y_pred': y_pred})
            test_acc = result['acc']
            
            self.log('test_acc', test_acc)
            print(f"\nTest Accuracy: {test_acc:.4f}")
            
            self.test_preds.clear()
            self.test_labels.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_acc',
            }
        }


class ExperimentRunner:
    def __init__(
        self,
        text_file_path: str,
        output_dir: str = './embeddings',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.text_file_path = text_file_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        self.text_processor = TextDataProcessor(text_file_path)
        self.results = {}
    
    def generate_or_load_embeddings(
        self,
        model_name: str,
        force_regenerate: bool = False,
        **config_kwargs
    ) -> Tuple[np.ndarray, EmbeddingConfig]:
        
        # Create config
        config = EmbeddingConfig(model_name=model_name, **config_kwargs)
        
        # Create filename
        config_str = f"{model_name.replace('/', '_')}_{config.pooling_strategy}"
        if config.use_title_only:
            config_str += "_title_only"
        elif config.use_abstract_only:
            config_str += "_abstract_only"
        
        embeddings_path = self.output_dir / f"{config_str}.npy"
        config_path = self.output_dir / f"{config_str}_config.json"
        
        # Check if embeddings exist
        if embeddings_path.exists() and not force_regenerate:
            print(f"\nLoading cached embeddings from {embeddings_path}")
            embeddings = np.load(embeddings_path)
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
            print(f"Loaded embeddings shape: {embeddings.shape}")
            return embeddings, config
        
        # Generate embeddings
        print(f"\nGenerating embeddings with {model_name}")
        generator = TextEmbeddingGenerator(config, device=self.device)
        
        # Prepare texts
        texts = self.text_processor.prepare_texts(
            use_title_only=config.use_title_only,
            use_abstract_only=config.use_abstract_only
        )
        
        # Generate
        embeddings = generator.generate_embeddings(texts)
        
        # Save
        np.save(embeddings_path, embeddings)
        with open(config_path, 'w') as f:
            json.dump(config.__dict__, f, indent=2)
        
        print(f"Saved embeddings to {embeddings_path}")
        print(f"Embeddings shape: {embeddings.shape}")
        
        return embeddings, config
    
    def run_single_experiment(
        self,
        embeddings: np.ndarray,
        model_type: str,  # 'mlp', 'sage', 'gat', 'gcn'
        experiment_name: str,
        hidden_channels: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        lr: float = 0.01,
        batch_size: int = 1024,
        max_epochs: int = 100,
        use_wandb: bool = False,
    ):
        
        print(f"\n{'='*60}")
        print(f"Running: {experiment_name}")
        print(f"Model: {model_type.upper()}")
        print(f"{'='*60}\n")
        
        # Setup data
        datamodule = OGBArxivWithEmbeddingsDataModule(
            embeddings=embeddings,
            batch_size=batch_size,
            num_neighbors=[10, 10],
        )
        datamodule.setup()
        
        # Create model
        if model_type.lower() == 'mlp':
            model = MLPClassifier(
                in_channels=datamodule.num_features,
                hidden_channels=hidden_channels,
                num_classes=datamodule.num_classes,
                num_layers=num_layers,
                dropout=dropout,
                lr=lr,
            )
        else:
            model = GNNClassifier(
                gnn_type=model_type,
                in_channels=datamodule.num_features,
                hidden_channels=hidden_channels,
                num_classes=datamodule.num_classes,
                num_layers=num_layers,
                dropout=dropout,
                lr=lr,
            )
        
        # Setup logger
        if use_wandb:
            logger = WandbLogger(
                project="ogbn-arxiv-text-embeddings",
                name=experiment_name,
                log_model=False,
            )
        else:
            logger = TensorBoardLogger(
                "lightning_logs",
                name=experiment_name
            )
        
        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor='val_acc',
            mode='max',
            save_top_k=1,
            filename='{epoch}-{val_acc:.4f}',
        )
        
        early_stop_callback = EarlyStopping(
            monitor='val_acc',
            patience=20,
            mode='max',
            verbose=True,
        )
        
        # Trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator='auto',
            devices=1,
            logger=logger,
            callbacks=[checkpoint_callback, early_stop_callback],
            log_every_n_steps=10,
        )
        
        # Train
        trainer.fit(model, datamodule)
        
        # Test
        test_results = trainer.test(model, datamodule, ckpt_path='best')
        
        return {
            'trainer': trainer,
            'model': model,
            'test_acc': test_results[0]['test_acc'],
            'best_val_acc': checkpoint_callback.best_model_score.item(),
        }
    
    def run_full_comparison(
        self,
        embedding_models: List[str] = None,
        gnn_models: List[str] = None,
        pooling_strategies: List[str] = None,
        **train_kwargs
    ):
        
        if embedding_models is None:
            embedding_models = [
                'roberta-base',
                'bert-base-uncased',
                'distilbert-base-uncased',
                'allenai/scibert_scivocab_uncased',  # Scientific papers
                'sentence-transformers/all-MiniLM-L6-v2',  # Sentence embeddings
            ]
        
        if gnn_models is None:
            gnn_models = ['mlp', 'gcn', 'sage', 'gat']
        
        if pooling_strategies is None:
            pooling_strategies = ['cls', 'mean']
        
        all_results = {}
        
        for emb_model in embedding_models:
            for pooling in pooling_strategies:
                # Generate embeddings
                embeddings, config = self.generate_or_load_embeddings(
                    model_name=emb_model,
                    pooling_strategy=pooling,
                )
                
                emb_key = f"{emb_model.split('/')[-1]}_{pooling}"
                all_results[emb_key] = {}
                
                # Train with each GNN model
                for gnn_model in gnn_models:
                    exp_name = f"{emb_key}_{gnn_model}"
                    
                    result = self.run_single_experiment(
                        embeddings=embeddings,
                        model_type=gnn_model,
                        experiment_name=exp_name,
                        **train_kwargs
                    )
                    
                    all_results[emb_key][gnn_model] = result
        
        # Save results summary
        self.save_results_summary(all_results)
        
        return all_results
    
    def save_results_summary(self, results: Dict):
        summary = []
        
        for emb_model, gnn_results in results.items():
            for gnn_model, result in gnn_results.items():
                summary.append({
                    'embedding_model': emb_model,
                    'gnn_model': gnn_model,
                    'test_acc': result['test_acc'],
                    'best_val_acc': result['best_val_acc'],
                })
        
        df = pd.DataFrame(summary)
        df = df.sort_values('test_acc', ascending=False)
        
        # Save to CSV
        results_path = self.output_dir / 'results_summary.csv'
        df.to_csv(results_path, index=False)
        
        # Print summary
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(df.to_string(index=False))
        print("\n" + "="*80)
        
        return df


if __name__ == "__main__":
    # Configuration
    TEXT_FILE_PATH = "ogbn_arxiv_texts.tsv"
    
    # Initialize experiment runner
    runner = ExperimentRunner(
        text_file_path=TEXT_FILE_PATH,
        output_dir='./embeddings',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Run full comparison
    results = runner.run_full_comparison(
        embedding_models=[
            'roberta-base',
            'bert-base-uncased',
            'allenai/scibert_scivocab_uncased',
            'sentence-transformers/all-MiniLM-L6-v2',
        ],
        gnn_models=['mlp', 'gcn', 'sage', 'gat'],
        pooling_strategies=['cls', 'mean'],
        hidden_channels=256,
        num_layers=2,
        dropout=0.5,
        lr=0.01,
        batch_size=1024,
        max_epochs=100,
        use_wandb=False,
    )