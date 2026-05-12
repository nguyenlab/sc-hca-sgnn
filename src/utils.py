import os
import sys
import re
import solcx


def ASTExtractor(contract_bytes: bytes):
    solidity_version = getSolidityVersion(contract_bytes.decode("utf-8", errors="ignore"))
    solcx.install_solc(solidity_version)    
    solcx.set_solc_version(solidity_version)

    # compile the contract and return only the AST
    compiled_sol = solcx.compile_source(
        contract_bytes.decode("utf-8", errors="ignore"),
        output_values=["ast"],
        solc_version=solidity_version,
    )
    ast = compiled_sol[list(compiled_sol.keys())[0]]["ast"]
    return ast

def getSolidityVersion(source_code):
    # Match all pragma solidity statements (there may be multiple)
    pattern = r'pragma\s+solidity\s+([\^>=<\.\d\s]+);'
    matches = re.findall(pattern, source_code)
    
    if not matches:
        return None
    
    # Parse all version constraints and find the maximum (most restrictive)
    versions = []
    for version_constraint in matches:
        version = _parseVersionConstraint(version_constraint.strip())
        if version:
            versions.append(version)
    
    if not versions:
        return None
    
    # Return the highest version to satisfy all constraints
    # Sort and take the maximum
    sorted_versions = sorted(versions, key=lambda v: [int(x) for x in v.split('.')])
    selected_version = sorted_versions[-1]
    
    # Enforce minimum supported version (0.4.11)
    version_parts = [int(x) for x in selected_version.split('.')]
    if version_parts[0] == 0 and version_parts[1] == 4 and version_parts[2] < 11:
        # Upgrade to 0.4.11 (minimum supported by py-solc-x)
        selected_version = "0.4.11"
    
    # Check for language features that require specific versions
    # and upgrade if necessary
    min_version_needed = _detectMinimumVersionFromFeatures(source_code)
    if min_version_needed:
        selected_parts = [int(x) for x in selected_version.split('.')]
        needed_parts = [int(x) for x in min_version_needed.split('.')]
        
        # Compare versions and use the higher one
        if needed_parts > selected_parts:
            selected_version = min_version_needed
    
    return selected_version


def _detectMinimumVersionFromFeatures(source_code):
    # view/pure keywords introduced in 0.4.16
    if re.search(r'\b(view|pure)\b', source_code):
        return "0.4.16"
    
    # constructor keyword introduced in 0.4.22
    if re.search(r'\bconstructor\s*\(', source_code):
        return "0.4.22"
    
    # emit keyword for events introduced in 0.4.21
    if re.search(r'\bemit\s+\w+', source_code):
        return "0.4.21"
    
    return None


def _parseVersionConstraint(constraint):
    constraint = constraint.strip()
    
    # Extract all version numbers from the constraint
    versions = re.findall(r'(\d+\.\d+\.\d+|\d+\.\d+)', constraint)
    
    if not versions:
        return None
    
    # If it's a fixed version (no operators or only =)
    if not re.search(r'[\^>=<]', constraint):
        return versions[0]
    
    # If it starts with ^ (caret) - compatible with version
    if constraint.startswith('^'):
        return versions[0]
    
    # If it starts with ~ (tilde) - allows patch-level changes
    if constraint.startswith('~'):
        return versions[0]
    
    # For >= or > operators, return the first version
    if re.match(r'^[>=<]+', constraint):
        return versions[0]
    
    # For range constraints like ">=0.4.21 <0.6.0", return the first (minimum)
    if len(versions) > 1:
        return versions[0]
    
    return versions[0] if versions else None


     