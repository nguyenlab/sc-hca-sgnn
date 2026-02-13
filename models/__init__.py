

# ============================================================================
# Data Loading Classes (shared)
# ============================================================================
from models.data import (
    ContractGraphDataset,
    create_data_loaders,
    create_data_loaders_with_irl_test,
    get_input_dim,
    get_num_classes,
    get_num_edge_types,
    VULN_TYPES,
    VULN_TO_IDX,
    IDX_TO_VULN,
    NODE_TYPES,
    NUM_NODE_TYPES,
    NUM_EDGE_TYPES,
)

from models.function_level_data import (
    FunctionLevelDataset,
    FunctionLevelData,
    FunctionLevelStats,
    create_function_level_loaders,
    create_function_level_loaders_with_irl_test,
    pool_by_function,
    json_to_function_level_data,
)

from models.bsgvd_data import (
    BSGVDDataset,
    BSGVDDataStats,
    create_bsgvd_loaders,
    create_bsgvd_loaders_with_irl_test,
    get_bsgvd_input_dim,
    compute_bsgvd_class_weights,
)

# ============================================================================
# Graph-Level Models
# ============================================================================
from models.graph_level.gcn import (
    GCNEncoder,
    NodeLevelGCN,
    GraphLevelGCN,
    CombinedGCN,
)

from models.graph_level.multi_edge_gcn import (
    create_model,
    GraphLevelMultiEdgeGCN,
    HierarchicalMultiEdgeGCN,
)

from models.graph_level.dr_gcn import (
    create_dr_gcn_model,
    create_tmp_model,
    DRGCNClassifier,
    TMPClassifier,
    DRGCNConv,
    TMPConv,
    DRGCNEncoder,
    TMPEncoder,
)

from models.graph_level.attention_models import (
    create_gat_model,
    create_transformer_gnn_model,
    GATClassifier,
    TransformerGNNClassifier,
    GATEncoder,
    TransformerGNNEncoder,
)

from models.graph_level.bugsweeper import (
    create_bugsweeper_model,
    create_bugsweeper_light_model,
    BugSweeper,
    BugSweeperLight,
    CGNN,
    SecondStageGNN,
    CodeGraphPool,
)

from models.graph_level.scvhunter import (
    create_scvhunter_model,
    SCVHUNTERClassifier,
    SCVHUNTEREncoder,
    HeterogeneousAttentionLayer,
)

from models.graph_level.mlagnn import (
    create_mlagnn_model,
    MLAGNNClassifier,
    MLAGNNEncoder,
    AdaptiveAttentionConv,
)

from models.graph_level.bsgvd import (
    create_bsgvd_model,
    BSGVDClassifier,
    BSGVDEncoder,
    BSGVDFullBimodal,
    GATv2Branch,
)

from models.graph_level.fasttext_embedding import (
    FastTextEmbedder,
    FastTextFeatureExtractor,
    FastTextConfig,
    train_fasttext_on_solidity,
    load_fasttext_model,
    solidity_tokenizer,
    get_node_label,
    GENSIM_AVAILABLE,
)

# ============================================================================
# Subgraph-Level (Function-Level) Models
# ============================================================================
from models.subgraph_level.function_level_gnn import (
    create_function_level_model,
    create_function_level_model_with_graph,
    FunctionLevelGNN,
    FunctionLevelGNNWithGraphOutput,
    DualPathEncoder,
    FunctionPooling,
    FunctionLevelClassifier,
)

from models.subgraph_level.dr_gcn import (
    create_function_level_dr_gcn_model,
    create_function_level_tmp_model,
    FunctionLevelDRGCN,
    FunctionLevelTMP,
)

from models.subgraph_level.attention_models import (
    create_function_level_gat_model,
    create_function_level_transformer_model,
    FunctionLevelGAT,
    FunctionLevelTransformer,
)

from models.subgraph_level.scvhunter import (
    create_function_level_scvhunter_model,
    FunctionLevelSCVHUNTER,
)

from models.subgraph_level.mlagnn import (
    create_function_level_mlagnn_model,
    FunctionLevelMLAGNN,
)

from models.subgraph_level.hierarchical_cross_attention import (
    create_hca_sgnn_model,
    create_function_level_hca_model,
    HierarchicalCrossAttentionSGNN,
    EdgeTypeAwareConv,
    FunctionContractCrossAttention,
    HierarchicalFunctionPooling,
    GatedLocalGlobalFusion,
)

__all__ = [
    # =========== Data Classes ===========
    'ContractGraphDataset',
    'create_data_loaders',
    'create_data_loaders_with_irl_test',
    'get_input_dim',
    'get_num_classes',
    'get_num_edge_types',
    'VULN_TYPES',
    'VULN_TO_IDX',
    'IDX_TO_VULN',
    'NODE_TYPES',
    'NUM_NODE_TYPES',
    'NUM_EDGE_TYPES',
    # Function-level data
    'FunctionLevelDataset',
    'FunctionLevelData',
    'FunctionLevelStats',
    'create_function_level_loaders',
    'create_function_level_loaders_with_irl_test',
    'pool_by_function',
    'json_to_function_level_data',
    # BSGVD data (FastText embeddings)
    'BSGVDDataset',
    'BSGVDDataStats',
    'create_bsgvd_loaders',
    'create_bsgvd_loaders_with_irl_test',
    'get_bsgvd_input_dim',
    'compute_bsgvd_class_weights',
    
    # =========== Graph-Level Models ===========
    # GCN
    'GCNEncoder',
    'NodeLevelGCN',
    'GraphLevelGCN',
    'CombinedGCN',
    # Multi-edge GCN
    'create_model',
    'GraphLevelMultiEdgeGCN',
    'HierarchicalMultiEdgeGCN',
    # DR-GCN / TMP
    'create_dr_gcn_model',
    'create_tmp_model',
    'DRGCNClassifier',
    'TMPClassifier',
    'DRGCNConv',
    'TMPConv',
    'DRGCNEncoder',
    'TMPEncoder',
    # Attention models
    'create_gat_model',
    'create_transformer_gnn_model',
    'GATClassifier',
    'TransformerGNNClassifier',
    'GATEncoder',
    'TransformerGNNEncoder',
    # BugSweeper
    'create_bugsweeper_model',
    'create_bugsweeper_light_model',
    'BugSweeper',
    'BugSweeperLight',
    'CGNN',
    'SecondStageGNN',
    'CodeGraphPool',
    # SCVHUNTER
    'create_scvhunter_model',
    'SCVHUNTERClassifier',
    'SCVHUNTEREncoder',
    'HeterogeneousAttentionLayer',
    # ML-AGNN
    'create_mlagnn_model',
    'MLAGNNClassifier',
    'MLAGNNEncoder',
    'AdaptiveAttentionConv',
    # BSGVD
    'create_bsgvd_model',
    'BSGVDClassifier',
    'BSGVDEncoder',
    'BSGVDFullBimodal',
    'GATv2Branch',
    # FastText Embeddings
    'FastTextEmbedder',
    'FastTextFeatureExtractor',
    'FastTextConfig',
    'train_fasttext_on_solidity',
    'load_fasttext_model',
    'solidity_tokenizer',
    'get_node_label',
    'GENSIM_AVAILABLE',
    
    # =========== Subgraph-Level (Function-Level) Models ===========
    # Base function-level GNN
    'create_function_level_model',
    'create_function_level_model_with_graph',
    'FunctionLevelGNN',
    'FunctionLevelGNNWithGraphOutput',
    'DualPathEncoder',
    'FunctionPooling',
    'FunctionLevelClassifier',
    # DR-GCN / TMP
    'create_function_level_dr_gcn_model',
    'create_function_level_tmp_model',
    'FunctionLevelDRGCN',
    'FunctionLevelTMP',
    # Attention models
    'create_function_level_gat_model',
    'create_function_level_transformer_model',
    'FunctionLevelGAT',
    'FunctionLevelTransformer',
    # SCVHUNTER
    'create_function_level_scvhunter_model',
    'FunctionLevelSCVHUNTER',
    # ML-AGNN
    'create_function_level_mlagnn_model',
    'FunctionLevelMLAGNN',
    # HCA-SGNN (Hierarchical Cross-Attention Subgraph GNN)
    'create_hca_sgnn_model',
    'create_function_level_hca_model',
    'HierarchicalCrossAttentionSGNN',
    'EdgeTypeAwareConv',
    'FunctionContractCrossAttention',
    'HierarchicalFunctionPooling',
    'GatedLocalGlobalFusion',
]
