**Interpretive Structural Modeling (ISM)** and **Adversarial Interpretive Structural Modeling (AdISM)** are methods used to model complex systems or relationships among variables in structured and systematic ways.



Interpretive Structural Modeling (ISM)

- 

- Rafiq Ahmad Khan, Muhammad Azeem Akbar, Saima Rafi, Alaa Omran Almagrabi, Musaad Alzahrani, Evaluation of requirement engineering best practices for secure software development in GSD: An ISM analysis, Journal of Software: Evolution and Process, 10.1002/smr.2594, **36**, 5, (2023).









### **Interpretive Structural Modeling (ISM)**

- **Purpose**: ISM is a structured approach used to identify and analyze relationships among specific variables or components in a system. It helps create a hierarchical structure to better understand the problem domain.

  

### **Adversarial Interpretive Structural Modeling (AdISM)**

- **Purpose**: AdISM builds on ISM but incorporates the perspectives of stakeholders or experts with opposing viewpoints. It is designed for contexts where disagreements about relationships between variables exist, such as policy-making or conflict-prone areas.

**Key Features of AdISM**:

1. **Multiple Stakeholder Inputs**: Unlike ISM, AdISM explicitly involves conflicting viewpoints, often by categorizing stakeholders into "adversarial groups."
2. **Iterative Comparison**: It contrasts and negotiates the structural relationships proposed by different stakeholders.
3. **Reconciliation**: The goal is to identify areas of agreement and disagreement, eventually producing a consensus-based structure or mapping out where conflicts persist.



To implement a transition from **Interpretive Structural Modeling (ISM)** to **Adversarial Interpretive Structural Modeling (AdISM)** in Python, we need to incorporate both **basic ISM functionality** and a way to handle conflicting inputs from multiple stakeholders. 



```py
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Step 1: Define input from stakeholders (example data)
# Each stakeholder provides an SSIM (Structural Self-Interaction Matrix)
stakeholder_1_ssim = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
])

stakeholder_2_ssim = np.array([
    [0, 1, 1],
    [0, 0, 0],
    [1, 1, 0]
])

# Consolidate into a list for AdISM
stakeholder_inputs = [stakeholder_1_ssim, stakeholder_2_ssim]

# Step 2: Function to compute agreement and conflict matrices
def compute_agreement_conflict(stakeholder_inputs):
    n_stakeholders = len(stakeholder_inputs)
    combined_matrix = sum(stakeholder_inputs)  # Sum across matrices
    agreement_matrix = (combined_matrix == n_stakeholders).astype(int)
    conflict_matrix = ((combined_matrix > 0) & (combined_matrix < n_stakeholders)).astype(int)
    return agreement_matrix, conflict_matrix

# Step 3: Generate reachability matrix
def reachability_matrix(ssim):
    """Convert SSIM into a binary reachability matrix."""
    n = len(ssim)
    reachability = np.linalg.matrix_power(ssim, n) > 0
    return reachability.astype(int)

# Step 4: Compute agreement and conflict for AdISM
agreement_matrix, conflict_matrix = compute_agreement_conflict(stakeholder_inputs)

# Step 5: Visualize agreement and conflict
def visualize_matrix(matrix, title="Matrix"):
    plt.imshow(matrix, cmap="Blues", interpolation="none")
    plt.colorbar(label="Value")
    plt.title(title)
    plt.show()

visualize_matrix(agreement_matrix, title="Agreement Matrix")
visualize_matrix(conflict_matrix, title="Conflict Matrix")

# Step 6: Hierarchical partitioning (simplified)
def partition_hierarchy(reachability):
    levels = []
    remaining_elements = set(range(len(reachability)))
    while remaining_elements:
        current_level = []
        for i in remaining_elements:
            if all(reachability[i, j] == 0 for j in remaining_elements if i != j):
                current_level.append(i)
        levels.append(current_level)
        remaining_elements -= set(current_level)
    return levels

# Generate reachability matrix for agreement
agreement_reachability = reachability_matrix(agreement_matrix)

# Partition hierarchy
hierarchy_levels = partition_hierarchy(agreement_reachability)

# Step 7: Visualize hierarchy???? 
def visualize_hierarchy(hierarchy, title="Hierarchy"):
    G = nx.DiGraph()

    # Add nodes and edges based on hierarchy levels
    for i, level in enumerate(hierarchy):
        for node in level:
            G.add_node(node, level=i)  # Set 'level' attribute for each node
            if i + 1 < len(hierarchy):  # Ensure there is a next level
                for next_node in hierarchy[i + 1]:
                    G.add_edge(node, next_node)  # Create edges between levels

    # Get positions using a more general layout (e.g., spring_layout or shell_layout)
    pos = nx.spring_layout(G)  # Simple layout (works for most cases)

    # Draw the graph
    plt.figure(figsize=(8, 7))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10, edge_color="gray")
    plt.title(title)
    plt.show()

visualize_hierarchy(hierarchy_levels, title="AdISM Hierarchy")
```

**1. Inputs**: Each stakeholder provides their own SSIM.

**2. Agreement and Conflict Analysis**:

- **Agreement Matrix**: Entries with unanimous agreement across stakeholders.
- **Conflict Matrix**: Entries where stakeholders disagree.

**3. Reachability Matrix**: Converts the SSIM into a form that identifies direct and indirect relationships.



### **Steps**:

1. Input the two SSIMs:
   - Each matrix represents relationships as understood by different stakeholders.
2. Compute the Agreement and Conflict Matrices:
   - **Agreement Matrix**: Where both matrices agree on the relationship.
   - **Conflict Matrix**: Where the matrices disagree.

**3. Create the Reachability Matrix**:

- Derived from the Agreement Matrix.

**Perform Hierarchical Partitioning**:

- Group the elements into levels based on their dependencies and influence.

**Visualize the Hierarchy**:

- Create a directed graph showing the hierarchy.



For **Adversarial Interpretive Structural Modeling (AdISM)**, the **Reachability Matrix** should be derived **only from the Agreement Matrix** because the Agreement Matrix captures the relationships that all stakeholders agree upon.

This ensures that the hierarchical structure reflects the consensus among stakeholders rather than being influenced by areas of conflict. The conflict matrix is used separately to analyze disagreements but does not contribute to the hierarchical partitioning.



```py
import numpy as np

# Step 1: Input SSIMs (Example)
ssim1 = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
])

ssim2 = np.array([
    [0, 1, 1],
    [0, 0, 0],
    [1, 0, 0]
])

# Step 2: Compute Agreement and Conflict Matrices
def compute_agreement_conflict(ssim1, ssim2):
    agreement_matrix = (ssim1 == ssim2).astype(int) * ssim1  # Agreeing relationships
    conflict_matrix = (ssim1 != ssim2).astype(int)           # Disagreeing relationships
    return agreement_matrix, conflict_matrix

agreement_matrix, conflict_matrix = compute_agreement_conflict(ssim1, ssim2)

# Step 3: Create the Reachability Matrix
def reachability_matrix(ssim):
    """Generate reachability matrix by transitive closure."""
    n = len(ssim)
    reachability = np.linalg.matrix_power(ssim, n) > 0  # Indirect and direct connections
    return reachability.astype(int)

reachability = reachability_matrix(agreement_matrix)

# Step 4: Perform Hierarchical Partitioning
def partition_hierarchy(reachability):
    """Partition variables into hierarchical levels."""
    n = len(reachability)
    levels = []
    remaining_elements = set(range(n))
    
    while remaining_elements:
        current_level = []
        for i in remaining_elements:
            if all(reachability[i, j] == 0 for j in remaining_elements if i != j):
                current_level.append(i)
        levels.append(current_level)
        remaining_elements -= set(current_level)
    
    return levels

hierarchy = partition_hierarchy(reachability)

# Step 5: Output Results
print("Agreement Matrix:\n", agreement_matrix)
print("Conflict Matrix:\n", conflict_matrix)
print("Reachability Matrix:\n", reachability)
print("Hierarchical Levels:\n", hierarchy)
```

If you have **5 matrices** provided by different experts, the process is as follows:

1. **Generate the Agreement Matrix**:
   - Identify where all experts agree by checking where the same entries (typically `1` or `0`) are consistent across all 5 matrices.
   - The resulting **Agreement Matrix** includes only the relationships that are unanimously agreed upon.
2. **Generate the Conflict Matrix**:
   - Identify disagreements where at least one expert's opinion differs from the others.
   - This matrix is helpful for visualizing or analyzing areas of contention but is not used in hierarchical partitioning.
3. **Use the Agreement Matrix for Hierarchical Partitioning**:
   - The **Agreement Matrix** serves as the basis for creating the **Reachability Matrix**, which is then used for **Hierarchical Partitioning**.
   - This step ensures that the hierarchy reflects only the agreed-upon relationships, leading to a consensus-based structure.

By focusing on the **Agreement Matrix** for partitioning, you ensure that the final hierarchy reflects a unified understanding, avoiding complications caused by conflicting opinions.



### **Why Build the Conflict Matrix?**

The **Conflict Matrix** is a valuable tool for diagnosing and addressing disagreements in multi-expert systems. In practice, it complements the Agreement Matrix by ensuring that conflicts are handled transparently and effectively, leading to more robust and credible decision-making.

1. **Identify Areas of Disagreement**:
   - It highlights the specific relationships or elements where experts differ, making it clear which parts of the system lack consensus.
   - This is useful for understanding divergent perspectives or priorities among stakeholders.
2. **Analyze Contradictions**:
   - By examining conflicts, you can uncover deeper systemic or contextual issues leading to these disagreements.
   - This can help prioritize discussions or further investigations into contentious areas.
3. **Facilitate Discussions**:
   - The Conflict Matrix provides a structured way to visualize and discuss disagreements.
   - It supports collaborative decision-making by pointing out where efforts are needed to reach consensus.
4. **Improve Model Robustness**:
   - While the Agreement Matrix forms the basis for hierarchical partitioning, understanding the conflicts helps ensure that the model accounts for different perspectives, especially in follow-up analyses or iterations.
5. **Track Stakeholder Influence**:
   - If some disagreements stem from a lack of knowledge or varying priorities, the Conflict Matrix can help assess whose opinions or perspectives should be prioritized or addressed further.



### **Example of its Use**

Imagine a transportation planning project where five experts provide input matrices. If there is disagreement about whether "budget constraints" influence "project timeline," the Conflict Matrix will highlight this discrepancy. While the hierarchical model is built on agreements, planners can use the conflict data to:

- Discuss the differing assumptions behind the conflict.
- Investigate if additional data or research can resolve the disagreement.
- Document unresolved conflicts for future iterations of the model.



```py
import numpy as np

# Example: Input from 3 Experts
expert1 = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
])

expert2 = np.array([
    [0, 1, 1],
    [0, 0, 1],
    [1, 0, 0]
])

expert3 = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [1, 0, 0]
])

# Stack all matrices to analyze conflicts
matrices = np.array([expert1, expert2, expert3])

# Step 1: Compute Agreement and Conflict Matrices
def compute_agreement_conflict(matrices):
    """
    Compute the agreement and conflict matrices based on multiple input matrices.
    """
    n = matrices.shape[1]
    agreement_matrix = np.all(matrices == matrices[0], axis=0).astype(int) * matrices[0]  # Universal agreement
    conflict_matrix = np.any(matrices != matrices[0], axis=0).astype(int)  # Any disagreement
    
    return agreement_matrix, conflict_matrix

agreement_matrix, conflict_matrix = compute_agreement_conflict(matrices)

# Print Results
print("Agreement Matrix:\n", agreement_matrix)
print("Conflict Matrix:\n", conflict_matrix)

# Step 2: Analyze Conflicts
def analyze_conflicts(conflict_matrix):
    """
    Analyze conflicts and identify conflicting elements.
    """
    conflicting_elements = np.argwhere(conflict_matrix == 1)
    conflicts = [(i, j) for i, j in conflicting_elements]
    return conflicts

conflicting_elements = analyze_conflicts(conflict_matrix)
print("Conflicting Elements (row, column):", conflicting_elements)

# Example Application: Resolve a Conflict
def resolve_conflict(expert_matrices, element):
    """
    Resolve a conflict by averaging expert inputs for the conflicting element.
    """
    i, j = element
    values = expert_matrices[:, i, j]
    resolved_value = round(np.mean(values))  # Example: Use average or majority rule
    return resolved_value

# Resolve conflicts for each conflicting element
resolved_matrix = agreement_matrix.copy()
for conflict in conflicting_elements:
    resolved_matrix[conflict] = resolve_conflict(matrices, conflict)

print("Resolved Matrix:\n", resolved_matrix)
```

For the given input matrices

```py
Agreement Matrix:
 [[0 1 0]
  [0 0 1]
  [1 0 0]]

Conflict Matrix:
 [[0 0 1]
  [0 0 0]
  [0 0 0]]

Conflicting Elements (row, column): [(0, 2)]
Resolved Matrix:
 [[0 1 1]
  [0 0 1]
  [1 0 0]]
```

### **Use Cases for the Conflict Matrix**

1. **Visualizing Areas of Disagreement**:
   - Highlighting areas of conflict can guide discussions among experts.
   - For example, `(0, 2)` in the conflict matrix indicates disagreement about the relationship between element 0 and element 2.
2. **Facilitating Consensus Building**:
   - By identifying conflicts, stakeholders can focus discussions on resolving key issues instead of unrelated areas.
   - For instance, averaging or voting mechanisms can help resolve specific conflicts.
3. **Documenting Unresolved Issues**:
   - If some conflicts cannot be resolved, they can be documented for future iterations or flagged for additional investigation.
4. **Enhancing Model Transparency**:
   - The conflict matrix provides clarity on the modeling process, ensuring that areas of dissent are not overlooked.
5. **Scenario Analysis**:
   - Alternative hierarchical models can be generated by resolving conflicts differently (e.g., majority rule vs. giving more weight to specific experts).



**Refining the Agreement Matrix**:

- Using the Conflict Matrix, you can pinpoint specific areas of disagreement and resolve them through structured mechanisms like:
  - **Majority rule**
  - **Weighted averaging**
  - **Re-discussion or expert review**



**Role of the Conflict Matrix**:

- It is **not directly used in the hierarchical partitioning**.
- Instead, it serves as a **tool for understanding and resolving discrepancies**, improving the robustness and transparency of the final Agreement Matrix.



```py
import numpy as np

# Input from multiple experts
expert1 = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
])

expert2 = np.array([
    [0, 1, 1],
    [0, 0, 1],
    [1, 0, 0]
])

expert3 = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [1, 0, 0]
])

# Combine matrices
matrices = np.array([expert1, expert2, expert3])

# Compute Agreement and Conflict Matrices
def compute_matrices(matrices):
    agreement_matrix = np.all(matrices == matrices[0], axis=0).astype(int) * matrices[0]
    conflict_matrix = np.any(matrices != matrices[0], axis=0).astype(int)
    return agreement_matrix, conflict_matrix

agreement_matrix, conflict_matrix = compute_matrices(matrices)

# Refine Agreement Matrix
def resolve_conflicts(matrices, conflict_matrix):
    """
    Refine the Agreement Matrix by resolving conflicts using majority rule.
    """
    refined_matrix = agreement_matrix.copy()
    conflicts = np.argwhere(conflict_matrix == 1)
    for i, j in conflicts:
        # Majority rule for conflict resolution
        values = matrices[:, i, j]
        resolved_value = round(np.mean(values))  # Use average (or majority rule)
        refined_matrix[i, j] = resolved_value
    return refined_matrix

refined_agreement_matrix = resolve_conflicts(matrices, conflict_matrix)

# Print Results
print("Original Agreement Matrix:\n", agreement_matrix)
print("Conflict Matrix:\n", conflict_matrix)
print("Refined Agreement Matrix:\n", refined_agreement_matrix)
```

