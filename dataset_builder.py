import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

from src.utils import ASTExtractor, getSolidityVersion


# Edge type constants
EDGE_TYPE_AST = 0       # Parent-child (structural)
EDGE_TYPE_REF = 1       # Reference/data flow (Identifier -> Declaration)
EDGE_TYPE_CFG_NEXT = 2  # Control flow: next statement
EDGE_TYPE_CFG_TRUE = 3  # Control flow: true branch
EDGE_TYPE_CFG_FALSE = 4 # Control flow: false branch
EDGE_TYPE_CALL = 5      # Function call (FunctionCall -> FunctionDefinition)
EDGE_TYPE_INHERIT = 6   # Inheritance (Contract -> BaseContract)
EDGE_TYPE_GUARD = 7     # Guard condition (require/assert -> guarded statements)

# Node type constants - comprehensive list of Solidity AST node types
# This matches the list in models/data.py for consistency
NODE_TYPES = [
    'Unknown', 'SourceUnit', 'PragmaDirective', 'ImportDirective',
    'ContractDefinition', 'InterfaceDefinition', 'LibraryDefinition',
    'InheritanceSpecifier', 'UsingForDirective', 'StructDefinition',
    'EnumDefinition', 'EnumValue', 'VariableDeclaration', 'FunctionDefinition',
    'ModifierDefinition', 'ModifierInvocation', 'EventDefinition', 'ErrorDefinition',
    'ParameterList', 'OverrideSpecifier', 'Block', 'PlaceholderStatement',
    'IfStatement', 'WhileStatement', 'ForStatement', 'DoWhileStatement',
    'Continue', 'Break', 'Return', 'Throw', 'EmitStatement', 'RevertStatement',
    'TryStatement', 'TryCatchClause', 'VariableDeclarationStatement',
    'ExpressionStatement', 'UncheckedBlock', 'Assignment', 'TupleExpression',
    'UnaryOperation', 'BinaryOperation', 'FunctionCall', 'FunctionCallOptions',
    'NewExpression', 'MemberAccess', 'IndexAccess', 'IndexRangeAccess',
    'Identifier', 'ElementaryTypeNameExpression', 'Literal', 'Conditional',
    'ElementaryTypeName', 'UserDefinedTypeName', 'FunctionTypeName', 'Mapping',
    'ArrayTypeName', 'InlineAssembly', 'YulBlock', 'YulVariableDeclaration',
    'YulAssignment', 'YulFunctionCall', 'YulIdentifier', 'YulLiteral',
    'YulExpressionStatement', 'YulIf', 'YulSwitch', 'YulCase', 'YulForLoop',
    'YulFunctionDefinition', 'YulTypedName', 'UserDefinedValueTypeDefinition',
    # Elementary types (from legacy AST format)
    'uint', 'uint8', 'uint16', 'uint32', 'uint64', 'uint128', 'uint256',
    'int', 'int8', 'int16', 'int32', 'int64', 'int128', 'int256',
    'bool', 'address', 'bytes', 'bytes1', 'bytes2', 'bytes4', 'bytes8',
    'bytes16', 'bytes32', 'string', 'fixed', 'ufixed',
]
NODE_TYPE_TO_IDX = {n: i for i, n in enumerate(NODE_TYPES)}


def get_node_type_idx(node_type: str) -> int:
    """Get numeric index for a node type, defaulting to 0 (Unknown) for unrecognized types."""
    return NODE_TYPE_TO_IDX.get(node_type, 0)


@dataclass
class VulnerableRegion:
    """Represents an injected vulnerable region in the contract."""
    start_byte: int
    end_byte: int
    component: str  # 'vulnerable_code', 'state', etc.
    description: str
    
    def contains_byte(self, byte_position: int) -> bool:
        """Check if a byte position is within this region."""
        return self.start_byte <= byte_position < self.end_byte


@dataclass
class ContractMetadata:
    """Metadata for an injected vulnerable contract."""
    source_contract: str
    output_contract: str
    vulnerability_type: str
    injection_mode: str
    template_name: str
    solidity_version: str
    injected_regions: List[VulnerableRegion]
    
    @classmethod
    def from_json(cls, json_path: Path) -> 'ContractMetadata':
        """Load metadata from a JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        regions = [
            VulnerableRegion(
                start_byte=r['start_byte'],
                end_byte=r['end_byte'],
                component=r['component'],
                description=r['description']
            )
            for r in data.get('injected_regions', [])
        ]
        
        return cls(
            source_contract=data['source_contract'],
            output_contract=data['output_contract'],
            vulnerability_type=data['vulnerability_type'],
            injection_mode=data['injection_mode'],
            template_name=data['template_name'],
            solidity_version=data['solidity_version'],
            injected_regions=regions
        )


@dataclass
class ASTNode:
    """Simplified AST node for GNN processing."""
    node_id: int  # Our internal sequential ID
    ast_id: Optional[int]  # Original AST id from solc (for reference resolution)
    node_type: str
    node_type_idx: int  # Numeric index for node type (for efficient GNN features)
    src: str  # Source location in format "start:length:file_index"
    start_byte: int
    end_byte: int
    is_vulnerable: bool = False
    vulnerability_type: Optional[str] = None
    component: Optional[str] = None  # Which component of the vulnerability
    children: List[int] = field(default_factory=list)
    referenced_declaration: Optional[int] = None  # AST id this node references
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class ContractAST:
    """Processed AST with vulnerability labels for a contract."""
    contract_path: str
    vulnerability_type: str
    injection_mode: str
    template_name: str
    solidity_version: str
    nodes: List[ASTNode]
    edges: List[Tuple[int, int, int]]  # (from_id, to_id, edge_type)
    vulnerable_node_ids: List[int]
    edge_counts: Dict[str, int] = field(default_factory=dict)
    
    @property
    def num_ast_edges(self) -> int:
        return self.edge_counts.get('ast', 0)
    
    @property
    def num_ref_edges(self) -> int:
        return self.edge_counts.get('ref', 0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'contract_path': self.contract_path,
            'vulnerability_type': self.vulnerability_type,
            'injection_mode': self.injection_mode,
            'template_name': self.template_name,
            'solidity_version': self.solidity_version,
            'nodes': [
                {
                    'node_id': n.node_id,
                    'ast_id': n.ast_id,
                    'node_type': n.node_type,
                    'node_type_idx': n.node_type_idx,
                    'src': n.src,
                    'start_byte': n.start_byte,
                    'end_byte': n.end_byte,
                    'is_vulnerable': n.is_vulnerable,
                    'vulnerability_type': n.vulnerability_type,
                    'component': n.component,
                    'children': n.children,
                    'referenced_declaration': n.referenced_declaration,
                    'attributes': n.attributes
                }
                for n in self.nodes
            ],
            'edges': self.edges,
            'vulnerable_node_ids': self.vulnerable_node_ids,
            'edge_counts': self.edge_counts,
        }


class ASTProcessor:
    """Processes Solidity AST and labels vulnerable nodes with multi-edge support."""
    
    def __init__(self):
        self.node_counter = 0
        self.nodes: List[ASTNode] = []
        self.ast_edges: List[Tuple[int, int]] = []  # (parent, child) - structural
        self.ref_edges: List[Tuple[int, int]] = []  # (usage, declaration) - data flow
        self.cfg_next_edges: List[Tuple[int, int]] = []  # control flow: sequential
        self.cfg_true_edges: List[Tuple[int, int]] = []  # control flow: true branch
        self.cfg_false_edges: List[Tuple[int, int]] = []  # control flow: false branch
        self.call_edges: List[Tuple[int, int]] = []  # function call edges
        self.inherit_edges: List[Tuple[int, int]] = []  # inheritance edges
        self.guard_edges: List[Tuple[int, int]] = []  # guard condition edges
        self.vulnerable_node_ids: List[int] = []
        
        # Mapping from AST id (solc) to our internal node_id
        self.ast_id_to_node_id: Dict[int, int] = {}
        # Pending reference edges (we resolve them after full traversal)
        self.pending_refs: List[Tuple[int, int]] = []  # (our_node_id, ast_id_of_declaration)
        # Pending call edges
        self.pending_calls: List[Tuple[int, int]] = []  # (call_node_id, ast_id_of_function)
        # Pending inheritance edges
        self.pending_inherits: List[Tuple[int, int]] = []  # (contract_node_id, ast_id_of_base)
        
    def reset(self):
        """Reset processor state for a new contract."""
        self.node_counter = 0
        self.nodes = []
        self.ast_edges = []
        self.ref_edges = []
        self.cfg_next_edges = []
        self.cfg_true_edges = []
        self.cfg_false_edges = []
        self.call_edges = []
        self.inherit_edges = []
        self.guard_edges = []
        self.vulnerable_node_ids = []
        self.ast_id_to_node_id = {}
        self.pending_refs = []
        self.pending_calls = []
        self.pending_inherits = []
    
    def parse_src_location(self, src: str) -> Tuple[int, int]:
        """Parse source location string to get byte range."""
        if not src:
            return 0, 0
        parts = src.split(':')
        if len(parts) >= 2:
            start = int(parts[0])
            length = int(parts[1])
            return start, start + length
        return 0, 0
    
    def process_ast(
        self, 
        ast: Dict[str, Any], 
        vulnerable_regions: List[VulnerableRegion],
        vulnerability_type: str
    ) -> None:
        """
        Process the AST tree and label vulnerable nodes.
        
        Args:
            ast: The raw AST from solc compiler
            vulnerable_regions: List of vulnerable byte regions
            vulnerability_type: Type of vulnerability
        """
        # First pass: traverse AST and collect nodes/edges
        self._traverse_ast(ast, parent_id=None, vulnerable_regions=vulnerable_regions, vulnerability_type=vulnerability_type)
        # Second pass: resolve reference edges
        self._resolve_references()
    
    def _traverse_ast(
        self, 
        node: Dict[str, Any], 
        parent_id: Optional[int],
        vulnerable_regions: List[VulnerableRegion],
        vulnerability_type: str
    ) -> int:
        """Recursively traverse AST and create nodes."""
        node_id = self.node_counter
        self.node_counter += 1
        
        # Extract node type (handle both legacy and new AST formats)
        node_type = node.get('nodeType', node.get('name', 'Unknown'))
        
        # Extract AST id (if present)
        ast_id = node.get('id')
        if ast_id is not None:
            self.ast_id_to_node_id[ast_id] = node_id
        
        # Extract source location
        src = node.get('src', '')
        start_byte, end_byte = self.parse_src_location(src)
        
        # Check if this node is in a vulnerable region
        is_vulnerable = False
        component = None
        for region in vulnerable_regions:
            # A node is vulnerable if its byte range overlaps with a vulnerable region
            if self._ranges_overlap(start_byte, end_byte, region.start_byte, region.end_byte):
                is_vulnerable = True
                component = region.component
                break
        
        # Extract attributes (check both root level and 'attributes' dict for legacy format)
        attributes = {}
        referenced_declaration = None
        
        # Handle legacy AST format (attributes in 'attributes' dict)
        if 'attributes' in node and isinstance(node['attributes'], dict):
            attrs = node['attributes']
            referenced_declaration = attrs.get('referencedDeclaration')
            for key, value in attrs.items():
                if key not in ['referencedDeclaration', 'overloadedDeclarations']:
                    if isinstance(value, (str, int, float, bool, type(None))):
                        attributes[key] = value
        
        # Handle modern AST format (attributes at root level)
        for key, value in node.items():
            if key not in ['nodes', 'statements', 'body', 'members', 'children', 
                          'src', 'nodeType', 'name', 'id', 'attributes',
                          'expression', 'leftHandSide', 'rightHandSide',
                          'condition', 'trueBody', 'falseBody', 'initializationExpression',
                          'loopExpression', 'arguments', 'components', 'parameters',
                          'returnParameters', 'modifiers', 'baseContracts',
                          'subExpression', 'baseExpression', 'indexExpression']:
                if key == 'referencedDeclaration':
                    referenced_declaration = value
                elif isinstance(value, (str, int, float, bool, type(None))):
                    attributes[key] = value
                elif isinstance(value, dict) and 'nodeType' not in value and 'name' not in value:
                    # Include simple nested dicts (like typeName info)
                    attributes[key] = self._extract_simple_attrs(value)
        
        # Create AST node with node type index for efficient feature encoding
        ast_node = ASTNode(
            node_id=node_id,
            ast_id=ast_id,
            node_type=node_type,
            node_type_idx=get_node_type_idx(node_type),
            src=src,
            start_byte=start_byte,
            end_byte=end_byte,
            is_vulnerable=is_vulnerable,
            vulnerability_type=vulnerability_type if is_vulnerable else None,
            component=component,
            referenced_declaration=referenced_declaration,
            attributes=attributes
        )
        
        self.nodes.append(ast_node)
        
        if is_vulnerable:
            self.vulnerable_node_ids.append(node_id)
        
        # Add AST edge from parent (structural edge)
        if parent_id is not None:
            self.ast_edges.append((parent_id, node_id))
            self.nodes[parent_id].children.append(node_id)
        
        # Queue reference edge for later resolution
        if referenced_declaration is not None:
            self.pending_refs.append((node_id, referenced_declaration))
        
        # Extract additional semantic edges based on node type
        self._extract_semantic_edges(node, node_id, node_type)
        
        # Process children and extract control flow
        self._process_children_with_cfg(node, node_id, node_type, vulnerable_regions, vulnerability_type)
        
        return node_id
    
    def _extract_semantic_edges(self, node: Dict[str, Any], node_id: int, node_type: str) -> None:
        """Extract semantic edges based on node type."""
        # Inheritance edges: ContractDefinition -> base contracts
        if node_type == 'ContractDefinition':
            base_contracts = node.get('baseContracts', [])
            for base in base_contracts:
                if isinstance(base, dict):
                    # Navigate to UserDefinedTypeName to get referencedDeclaration
                    base_name = base.get('baseName', {})
                    if isinstance(base_name, dict):
                        ref_decl = base_name.get('referencedDeclaration')
                        if ref_decl:
                            self.pending_inherits.append((node_id, ref_decl))
        
        # Call edges: MemberAccess with function reference -> target function
        # (FunctionCall itself doesn't have referencedDeclaration, but its expression does)
        if node_type == 'MemberAccess':
            ref_decl = node.get('referencedDeclaration')
            attrs = node.get('attributes', {})
            if isinstance(attrs, dict):
                ref_decl = ref_decl or attrs.get('referencedDeclaration')
                member_type = attrs.get('type', '')
                # If it's a function type, add as potential call edge
                if ref_decl and 'function' in str(member_type):
                    self.pending_calls.append((node_id, ref_decl))
    
    def _process_children_with_cfg(
        self, 
        node: Dict[str, Any], 
        parent_id: int,
        node_type: str,
        vulnerable_regions: List[VulnerableRegion],
        vulnerability_type: str
    ) -> None:
        """Process children and extract control flow edges."""
        
        # Handle control flow for specific node types
        if node_type == 'Block':
            # Sequential statements in a block
            # Check both 'statements' (modern) and 'children' (legacy) formats
            statements = node.get('statements', node.get('children', []))
            if isinstance(statements, list):
                prev_stmt_id = None
                for stmt in statements:
                    if isinstance(stmt, dict) and ('nodeType' in stmt or 'name' in stmt):
                        stmt_id = self._traverse_ast(stmt, parent_id, vulnerable_regions, vulnerability_type)
                        if prev_stmt_id is not None:
                            self.cfg_next_edges.append((prev_stmt_id, stmt_id))
                        prev_stmt_id = stmt_id
                return  # Already processed children
        
        elif node_type == 'IfStatement':
            # Condition -> true branch, condition -> false branch
            condition = node.get('condition')
            true_body = node.get('trueBody')
            false_body = node.get('falseBody')
            
            # Legacy format uses 'children' array: [condition, trueBody, falseBody?]
            if not condition and 'children' in node:
                children = node.get('children', [])
                if len(children) >= 1:
                    condition = children[0]
                if len(children) >= 2:
                    true_body = children[1]
                if len(children) >= 3:
                    false_body = children[2]
            
            cond_id = None
            if condition and isinstance(condition, dict):
                cond_id = self._traverse_ast(condition, parent_id, vulnerable_regions, vulnerability_type)
            
            if true_body and isinstance(true_body, dict):
                true_id = self._traverse_ast(true_body, parent_id, vulnerable_regions, vulnerability_type)
                if cond_id is not None:
                    self.cfg_true_edges.append((cond_id, true_id))
            
            if false_body and isinstance(false_body, dict):
                false_id = self._traverse_ast(false_body, parent_id, vulnerable_regions, vulnerability_type)
                if cond_id is not None:
                    self.cfg_false_edges.append((cond_id, false_id))
            return  # Already processed children
        
        elif node_type == 'ForStatement' or node_type == 'WhileStatement':
            # Loop control flow
            condition = node.get('condition')
            body = node.get('body')
            
            # Legacy format
            if not condition and 'children' in node:
                children = node.get('children', [])
                # For ForStatement: [init, condition, loopExpr, body]
                # For WhileStatement: [condition, body]
                if node_type == 'WhileStatement' and len(children) >= 2:
                    condition = children[0]
                    body = children[1]
            
            cond_id = None
            if condition and isinstance(condition, dict):
                cond_id = self._traverse_ast(condition, parent_id, vulnerable_regions, vulnerability_type)
            
            if body and isinstance(body, dict):
                body_id = self._traverse_ast(body, parent_id, vulnerable_regions, vulnerability_type)
                if cond_id is not None:
                    self.cfg_true_edges.append((cond_id, body_id))
            
            # Process remaining children
            self._process_children(node, parent_id, vulnerable_regions, vulnerability_type)
            return
        
        elif node_type in ('FunctionCall',):
            # Check if this is a require/assert call (guard)
            expr = node.get('expression', {})
            if isinstance(expr, dict):
                expr_name = expr.get('name', '')
                if expr_name in ('require', 'assert', 'revert'):
                    # This is a guard - we'll link it to sibling statements later
                    # For now, just mark it (guard edges are complex to extract properly)
                    pass
        
        # Default: process all children normally
        self._process_children(node, parent_id, vulnerable_regions, vulnerability_type)
    
    def _process_children(
        self, 
        node: Dict[str, Any], 
        parent_id: int,
        vulnerable_regions: List[VulnerableRegion],
        vulnerability_type: str
    ) -> None:
        """Process all children of a node."""
        # Common child field names in Solidity AST
        child_fields = ['nodes', 'statements', 'body', 'members', 'children', 
                       'expression', 'leftHandSide', 'rightHandSide', 
                       'condition', 'trueBody', 'falseBody', 'initializationExpression',
                       'loopExpression', 'arguments', 'components', 'parameters',
                       'returnParameters', 'modifiers', 'baseContracts',
                       'subExpression', 'baseExpression', 'indexExpression']
        
        for field in child_fields:
            if field in node:
                child = node[field]
                if isinstance(child, list):
                    for item in child:
                        if isinstance(item, dict) and ('nodeType' in item or 'name' in item):
                            self._traverse_ast(item, parent_id, vulnerable_regions, vulnerability_type)
                elif isinstance(child, dict) and ('nodeType' in child or 'name' in child):
                    self._traverse_ast(child, parent_id, vulnerable_regions, vulnerability_type)
    
    def _ranges_overlap(self, start1: int, end1: int, start2: int, end2: int) -> bool:
        """Check if two byte ranges overlap."""
        return start1 < end2 and start2 < end1
    
    def _extract_simple_attrs(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Extract simple attributes from a nested dict."""
        result = {}
        for k, v in d.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                result[k] = v
        return result
    
    def _resolve_references(self) -> None:
        """Resolve pending reference, call, and inheritance edges after full AST traversal."""
        # Resolve reference edges (Identifier -> Declaration)
        for our_node_id, ast_decl_id in self.pending_refs:
            target_node_id = self.ast_id_to_node_id.get(ast_decl_id)
            if target_node_id is not None and target_node_id != our_node_id:
                self.ref_edges.append((our_node_id, target_node_id))
        
        # Resolve call edges (FunctionCall -> FunctionDefinition)
        for call_node_id, ast_func_id in self.pending_calls:
            target_node_id = self.ast_id_to_node_id.get(ast_func_id)
            if target_node_id is not None:
                self.call_edges.append((call_node_id, target_node_id))
        
        # Resolve inheritance edges (Contract -> BaseContract)
        for contract_node_id, ast_base_id in self.pending_inherits:
            target_node_id = self.ast_id_to_node_id.get(ast_base_id)
            if target_node_id is not None:
                self.inherit_edges.append((contract_node_id, target_node_id))
    
    def get_all_edges(self) -> List[Tuple[int, int, int]]:
        """Get all edges with their types."""
        edges = []
        for parent, child in self.ast_edges:
            edges.append((parent, child, EDGE_TYPE_AST))
        for usage, decl in self.ref_edges:
            edges.append((usage, decl, EDGE_TYPE_REF))
        for src, dst in self.cfg_next_edges:
            edges.append((src, dst, EDGE_TYPE_CFG_NEXT))
        for src, dst in self.cfg_true_edges:
            edges.append((src, dst, EDGE_TYPE_CFG_TRUE))
        for src, dst in self.cfg_false_edges:
            edges.append((src, dst, EDGE_TYPE_CFG_FALSE))
        for call, func in self.call_edges:
            edges.append((call, func, EDGE_TYPE_CALL))
        for contract, base in self.inherit_edges:
            edges.append((contract, base, EDGE_TYPE_INHERIT))
        for guard, guarded in self.guard_edges:
            edges.append((guard, guarded, EDGE_TYPE_GUARD))
        return edges
    
    def get_edge_counts(self) -> Dict[str, int]:
        """Get counts for each edge type."""
        return {
            'ast': len(self.ast_edges),
            'ref': len(self.ref_edges),
            'cfg_next': len(self.cfg_next_edges),
            'cfg_true': len(self.cfg_true_edges),
            'cfg_false': len(self.cfg_false_edges),
            'call': len(self.call_edges),
            'inherit': len(self.inherit_edges),
            'guard': len(self.guard_edges),
        }


class DatasetBuilder:
    """
    Main class for building the vulnerability detection dataset.
    
    Reads injected contracts and their metadata, converts to AST,
    and labels vulnerable nodes for GNN training.
    Optionally includes clean (non-vulnerable) contracts for balanced training.
    """
    
    def __init__(self, input_dir: Path, output_dir: Path, clean_dir: Optional[Path] = None, max_clean: Optional[int] = None):
        """
        Initialize the dataset builder.
        
        Args:
            input_dir: Directory containing injected contracts and metadata
            output_dir: Directory to save processed AST data
            clean_dir: Optional directory containing clean contracts
            max_clean: Maximum number of clean contracts to include
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.clean_dir = Path(clean_dir) if clean_dir else None
        self.max_clean = max_clean
        self.processor = ASTProcessor()
        
        # Statistics
        self.stats = {
            'total_contracts': 0,
            'successful': 0,
            'failed': 0,
            'by_vulnerability_type': defaultdict(int),
            'by_injection_mode': defaultdict(int),
            'edge_counts': defaultdict(int),
            'errors': []
        }
    
    def discover_contracts(self) -> List[Path]:
        """
        Discover all metadata JSON files in the input directory.
        
        Returns:
            List of paths to metadata files
        """
        metadata_files = []
        
        # Search in root and subdirectories (point, coupled)
        for json_file in self.input_dir.rglob('*.json'):
            metadata_files.append(json_file)
        
        return sorted(metadata_files)
    
    def process_contract(self, metadata_path: Path) -> Optional[ContractAST]:
        """
        Process a single contract and its metadata.
        
        Args:
            metadata_path: Path to the metadata JSON file
            
        Returns:
            ContractAST if successful, None otherwise
        """
        try:
            # Load metadata
            metadata = ContractMetadata.from_json(metadata_path)
            
            # Get the contract file path (should be alongside the JSON)
            contract_path = metadata_path.with_suffix('.sol')
            if not contract_path.exists():
                # Try the output_contract path from metadata
                contract_path = Path(metadata.output_contract)
            
            if not contract_path.exists():
                raise FileNotFoundError(f"Contract file not found: {contract_path}")
            
            # Read contract source
            with open(contract_path, 'rb') as f:
                contract_bytes = f.read()
            
            # Extract AST
            ast = ASTExtractor(contract_bytes)
            
            # Process AST and label vulnerable nodes
            self.processor.reset()
            self.processor.process_ast(ast, metadata.injected_regions, metadata.vulnerability_type)
            
            # Create ContractAST
            all_edges = self.processor.get_all_edges()
            edge_counts = self.processor.get_edge_counts()
            
            contract_ast = ContractAST(
                contract_path=str(contract_path),
                vulnerability_type=metadata.vulnerability_type,
                injection_mode=metadata.injection_mode,
                template_name=metadata.template_name,
                solidity_version=metadata.solidity_version,
                nodes=self.processor.nodes.copy(),
                edges=all_edges,
                vulnerable_node_ids=self.processor.vulnerable_node_ids.copy(),
                edge_counts=edge_counts
            )
            
            return contract_ast
            
        except Exception as e:
            self.stats['errors'].append({
                'file': str(metadata_path),
                'error': str(e)
            })
            return None
    
    def process_clean_contract(self, contract_path: Path) -> Optional[ContractAST]:
        """
        Process a clean (non-vulnerable) contract without metadata.
        
        Args:
            contract_path: Path to the Solidity contract file
            
        Returns:
            ContractAST if successful, None otherwise
        """
        try:
            # Read contract source
            with open(contract_path, 'rb') as f:
                contract_bytes = f.read()
            
            # Get Solidity version from contract
            try:
                solidity_version = getSolidityVersion(contract_bytes)
            except:
                solidity_version = "unknown"
            
            # Extract AST
            ast = ASTExtractor(contract_bytes)
            
            # Process AST without any vulnerable regions (empty list)
            self.processor.reset()
            self.processor.process_ast(ast, [], 'clean')
            
            # Create ContractAST with 'clean' vulnerability type
            all_edges = self.processor.get_all_edges()
            edge_counts = self.processor.get_edge_counts()
            
            contract_ast = ContractAST(
                contract_path=str(contract_path),
                vulnerability_type='clean',
                injection_mode='none',
                template_name='none',
                solidity_version=solidity_version,
                nodes=self.processor.nodes.copy(),
                edges=all_edges,
                vulnerable_node_ids=[],  # No vulnerable nodes in clean contracts
                edge_counts=edge_counts
            )
            
            return contract_ast
            
        except Exception as e:
            self.stats['errors'].append({
                'file': str(contract_path),
                'error': str(e)
            })
            return None
    
    def discover_clean_contracts(self) -> List[Path]:
        """
        Discover all Solidity contract files in the clean contracts directory.
        
        Returns:
            List of paths to clean contract files
        """
        if not self.clean_dir or not self.clean_dir.exists():
            return []
        
        clean_files = list(self.clean_dir.glob('*.sol'))
        return sorted(clean_files)
    
    def build_dataset(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Build the complete dataset from all contracts.
        
        Args:
            verbose: Print progress information
            
        Returns:
            Dataset statistics
        """
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Discover vulnerable contracts
        metadata_files = self.discover_contracts()
        
        # Discover clean contracts
        clean_files = self.discover_clean_contracts()
        if self.max_clean and len(clean_files) > self.max_clean:
            clean_files = clean_files[:self.max_clean]
        
        self.stats['total_contracts'] = len(metadata_files) + len(clean_files)
        
        if verbose:
            print(f"Found {len(metadata_files)} vulnerable contracts to process")
            print(f"Found {len(clean_files)} clean contracts to process")
            print("=" * 60)
        
        # Process each vulnerable contract
        processed_asts = []
        
        if verbose and metadata_files:
            print("\n[Processing Vulnerable Contracts]")
        
        for i, metadata_path in enumerate(metadata_files):
            if verbose:
                print(f"[{i+1}/{len(metadata_files)}] Processing: {metadata_path.name}")
            
            contract_ast = self.process_contract(metadata_path)
            
            if contract_ast:
                self.stats['successful'] += 1
                self.stats['by_vulnerability_type'][contract_ast.vulnerability_type] += 1
                self.stats['by_injection_mode'][contract_ast.injection_mode] += 1
                for etype, count in contract_ast.edge_counts.items():
                    self.stats['edge_counts'][etype] += count
                processed_asts.append(contract_ast)
                
                if verbose:
                    ec = contract_ast.edge_counts
                    print(f"    ✓ Nodes: {len(contract_ast.nodes)}, "
                          f"Edges: {len(contract_ast.edges)} (AST:{ec.get('ast',0)} Ref:{ec.get('ref',0)} CFG:{ec.get('cfg_next',0)+ec.get('cfg_true',0)+ec.get('cfg_false',0)} Call:{ec.get('call',0)}), "
                          f"Vulnerable: {len(contract_ast.vulnerable_node_ids)}")
            else:
                self.stats['failed'] += 1
                if verbose:
                    print(f"    ✗ Failed")
        
        # Process clean contracts
        if clean_files:
            if verbose:
                print("\n[Processing Clean Contracts]")
            
            for i, clean_path in enumerate(clean_files):
                if verbose:
                    print(f"[{i+1}/{len(clean_files)}] Processing: {clean_path.name}")
                
                contract_ast = self.process_clean_contract(clean_path)
                
                if contract_ast:
                    self.stats['successful'] += 1
                    self.stats['by_vulnerability_type'][contract_ast.vulnerability_type] += 1
                    self.stats['by_injection_mode'][contract_ast.injection_mode] += 1
                    for etype, count in contract_ast.edge_counts.items():
                        self.stats['edge_counts'][etype] += count
                    processed_asts.append(contract_ast)
                    
                    if verbose:
                        ec = contract_ast.edge_counts
                        print(f"    ✓ Nodes: {len(contract_ast.nodes)}, "
                              f"Edges: {len(contract_ast.edges)} (AST:{ec.get('ast',0)} Ref:{ec.get('ref',0)} CFG:{ec.get('cfg_next',0)+ec.get('cfg_true',0)+ec.get('cfg_false',0)} Call:{ec.get('call',0)}), "
                              f"Clean")
                else:
                    self.stats['failed'] += 1
                    if verbose:
                        print(f"    ✗ Failed")
        
        # Save processed ASTs
        self._save_dataset(processed_asts)
        
        # Convert defaultdicts to regular dicts for JSON serialization
        self.stats['by_vulnerability_type'] = dict(self.stats['by_vulnerability_type'])
        self.stats['by_injection_mode'] = dict(self.stats['by_injection_mode'])
        self.stats['edge_counts'] = dict(self.stats['edge_counts'])
        
        # Save statistics
        stats_path = self.output_dir / 'dataset_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        if verbose:
            print("=" * 60)
            print(f"Dataset built successfully!")
            print(f"  Total: {self.stats['total_contracts']}")
            print(f"  Successful: {self.stats['successful']}")
            print(f"  Failed: {self.stats['failed']}")
            print(f"\nEdge Statistics:")
            ec = self.stats['edge_counts']
            total = sum(ec.values())
            print(f"  Total edges: {total}")
            for etype in ['ast', 'ref', 'cfg_next', 'cfg_true', 'cfg_false', 'call', 'inherit', 'guard']:
                if ec.get(etype, 0) > 0:
                    print(f"    {etype}: {ec[etype]} ({100*ec[etype]/total:.1f}%)")
            print(f"\nBy vulnerability type:")
            for vtype, count in self.stats['by_vulnerability_type'].items():
                print(f"  {vtype}: {count}")
            print(f"\nOutput saved to: {self.output_dir}")
        
        return self.stats
    
    def _save_dataset(self, processed_asts: List[ContractAST]) -> None:
        """Save processed ASTs to separate ast_data and graph_data directories."""
        # Create separate directories for AST data and graph data
        ast_dir = self.output_dir / 'ast_data'
        graph_dir = self.output_dir / 'graph_data'
        ast_dir.mkdir(exist_ok=True)
        graph_dir.mkdir(exist_ok=True)
        
        for ast in processed_asts:
            # Create filename from contract path
            contract_name = Path(ast.contract_path).stem
            
            # Save AST data (raw node information, independent of graph preprocessing)
            ast_data = {
                'contract_path': ast.contract_path,
                'vulnerability_type': ast.vulnerability_type,
                'injection_mode': ast.injection_mode,
                'template_name': ast.template_name,
                'solidity_version': ast.solidity_version,
                'nodes': [
                    {
                        'node_id': n.node_id,
                        'ast_id': n.ast_id,
                        'node_type': n.node_type,
                        'node_type_idx': n.node_type_idx,
                        'src': n.src,
                        'start_byte': n.start_byte,
                        'end_byte': n.end_byte,
                        'is_vulnerable': n.is_vulnerable,
                        'vulnerability_type': n.vulnerability_type,
                        'component': n.component,
                        'children': n.children,
                        'referenced_declaration': n.referenced_declaration,
                        'attributes': n.attributes
                    }
                    for n in ast.nodes
                ],
                'vulnerable_node_ids': ast.vulnerable_node_ids,
            }
            
            ast_path = ast_dir / f"{contract_name}.json"
            with open(ast_path, 'w') as f:
                json.dump(ast_data, f, indent=2)
            
            # Save graph data (edge structure for GNN, can be regenerated from AST)
            graph_data = {
                'contract_path': ast.contract_path,
                'vulnerability_type': ast.vulnerability_type,
                'num_nodes': len(ast.nodes),
                'edges': ast.edges,  # (from_id, to_id, edge_type)
                'edge_counts': ast.edge_counts,
                'vulnerable_node_ids': ast.vulnerable_node_ids,
                'node_types': [n.node_type for n in ast.nodes],
                'node_type_idx': [n.node_type_idx for n in ast.nodes],  # Numeric indices for efficient feature encoding
                'node_start_bytes': [n.start_byte for n in ast.nodes],  # Position features
                'node_end_bytes': [n.end_byte for n in ast.nodes],
                'node_is_vulnerable': [n.is_vulnerable for n in ast.nodes],
            }
            
            graph_path = graph_dir / f"{contract_name}.json"
            with open(graph_path, 'w') as f:
                json.dump(graph_data, f, indent=2)
        
        # Save a combined index file with summary info
        index = {
            'total_contracts': len(processed_asts),
            'contracts': [
                {
                    'path': ast.contract_path,
                    'vulnerability_type': ast.vulnerability_type,
                    'injection_mode': ast.injection_mode,
                    'template_name': ast.template_name,
                    'num_nodes': len(ast.nodes),
                    'num_edges': len(ast.edges),
                    'edge_counts': ast.edge_counts,
                    'num_vulnerable_nodes': len(ast.vulnerable_node_ids),
                    'ast_file': f"ast_data/{Path(ast.contract_path).stem}.json",
                    'graph_file': f"graph_data/{Path(ast.contract_path).stem}.json"
                }
                for ast in processed_asts
            ]
        }
        
        index_path = self.output_dir / 'dataset_index.json'
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        
        # Save node type and edge type mappings for reference
        mappings = {
            'node_types': NODE_TYPES,
            'node_type_to_idx': NODE_TYPE_TO_IDX,
            'num_node_types': len(NODE_TYPES),
            'edge_types': ['ast', 'ref', 'cfg_next', 'cfg_true', 'cfg_false', 'call', 'inherit', 'guard'],
            'num_edge_types': 8,
        }
        mappings_path = self.output_dir / 'type_mappings.json'
        with open(mappings_path, 'w') as f:
            json.dump(mappings, f, indent=2)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build dataset for GNN-based smart contract vulnerability detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dataset_builder.py --input data/synthetic/sc-source/vulnerable --output data/synthetic/ast_dataset
  python dataset_builder.py --input data/synthetic/sc-source/vulnerable --clean data/synthetic/sc-source/clean --output data/synthetic/ast_dataset
  python dataset_builder.py --input data/synthetic/sc-source/vulnerable --clean data/synthetic/sc-source/clean --max-clean 1000 --output data/synthetic/ast_dataset
"""
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input directory containing injected contracts and metadata"
    )
    parser.add_argument(
        "--output", "-o", 
        required=True,
        help="Output directory for processed AST data"
    )
    parser.add_argument(
        "--clean", "-c",
        default=None,
        help="Directory containing clean (non-vulnerable) contracts"
    )
    parser.add_argument(
        "--max-clean",
        type=int,
        default=None,
        help="Maximum number of clean contracts to include (for balancing)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_arguments()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    clean_dir = Path(args.clean) if args.clean else None
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1
    
    if clean_dir and not clean_dir.exists():
        print(f"Error: Clean contracts directory not found: {clean_dir}")
        return 1
    
    builder = DatasetBuilder(input_dir, output_dir, clean_dir, args.max_clean)
    stats = builder.build_dataset(verbose=not args.quiet)
    
    if stats['failed'] > 0:
        print(f"\nWarning: {stats['failed']} contracts failed to process.")
        if stats['errors']:
            print("Errors:")
            for err in stats['errors'][:5]:  # Show first 5 errors
                print(f"  - {err['file']}: {err['error'][:100]}")
            if len(stats['errors']) > 5:
                print(f"  ... and {len(stats['errors']) - 5} more")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
