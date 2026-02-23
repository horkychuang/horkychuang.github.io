---
marp: true
theme: gaia
color: #000
size: 16:9
colorSecondary: #333
backgroundColor: #fef9e7
backgroundImage: url('images/background_1.JPG')
footer: 'Designed by Wen-Bin 2026-02-05'
paginate: true

---

<!-- _class: lead -->

# Big Data and Society

**Class 021 Network Characteristic** 

**國企 Wen-Bin Chuang**
**2026-02-14**

![bg right fit](images\0820_3.jpg)

---

## Core Concepts
<style scoped>
section {
    font-size: 20px;
}
</style>
| Metric                     | What it captures                                             | Real-world analogy                       |
| -------------------------- | ------------------------------------------------------------ | ---------------------------------------- |
| **Degree Centrality**      | Number of direct connections                                 | "How many friends do you have?"          |
| **Betweenness Centrality** | How often a node lies on shortest paths between others       | "Are you a bridge between groups?"       |
| **Closeness Centrality**   | How close a node is to all others (inverse of avg. path length) | "How fast can you spread news?"          |
| **Eigenvector Centrality** | Connected to important nodes                                 | “Friends with the powerful”              |
| **PageRank**               | Eigenvector centrality with teleportation                    | Important + hard to reach by random walk |
| **Katz Centrality**        | Rewards proximity to many nodes (decaying with distance)     | --                                       |



----

## Application in Social Media Analysis
<style scoped>
section {
    font-size: 26px;
}
</style>
In social media analysis, `influencers`—often referred to as `Key Opinion Leaders (KOLs)`—are individuals or entities that wield significant influence over others within a network. 在社交媒體分析中，“影響者”（常被稱為“關鍵意見領袖”，即KOL）是指在網路中對他人具有顯著影響力的個人或實體。

They shape opinions, drive trends, and facilitate information dissemination. 他們能夠塑造觀點、引領潮流，並促進資訊傳播。 

Using graph theory, we model social media as a graph where users are nodes and interactions (e.g., follows, retweets, mentions) are edges. 利用圖論，我們可以將社交媒體建模為一張圖，其中用戶是節點，而互動行為（例如關注、轉發、提及）則構成邊。

---
<style scoped>
section {
    font-size: 25px;
}
</style>
Influencers are identified through **centrality analysis**, which quantifies a node's importance based on its position and connections. 影響者的識別通常通過**中心性分析**（centrality analysis）實現，該方法根據節點在網路中的位置及其連接關係來量化其重要性。 

This helps in applications like targeted marketing, viral campaign planning, misinformation detection, or understanding power dynamics in online communities. 這一技術可應用於精准行銷、病毒式傳播活動策劃、虛假資訊檢測，以及理解線上社群中的權力結構等場景。



---

**Influencer / Centrality Analysis**（關鍵節點識別)
<style scoped>
section {
    font-size: 25px;
}
</style>
- **Concept**：度中心性（Degree Centrality）、接近中心性（Closeness）、介數中心性（Betweenness）、特徵向量中心性（Eigenvector Centrality）
- **Application**：找出最具影響力的用戶（如 tread、FB KOL），用於行銷或輿情引導。
- **Example**：在轉發網路中，介數高的使用者可能是資訊傳播的“橋樑”。

---

Centrality measures reveal different aspects of influence:
<style scoped>
section {
    font-size: 25px;
}
</style>
- **Local influencers**: High `direct` connections (e.g., celebrities with many followers) 擁有大量**直接連接**（例如擁有眾多粉絲的名人）.

- **Global influencers**: `Bridge` disparate groups or control information flow (e.g., journalists connecting communities) 充當不同群體之間的**橋樑**，或掌控資訊流動（例如連接多個社群的記者）.

- **Prestige influencers**聲望影響者: Associated with other high-status nodes (e.g., endorsed by peers) 與其它高地位節點密切相關（例如獲得同行認可或背書的人).

---
<style scoped>
section {
    font-size: 25px;
}
</style>
**Katz** Centrality measures a node’s influence by counting the number of walks (of all lengths) emanating from it, with longer walks attenuated by a factor α < 1 通過計算從某節點出發的所有長度路徑（walks）數量來衡量其影響力，其中較長路徑的貢獻會按衰減因數 α（α < 1）進行折扣.

- It’s like a “smarter” version of degree centrality that rewards nodes connected to other influential nodes 它可視為度中心性（degree centrality）的一種“更智慧”版本，不僅考慮直接連接數量，還獎勵那些與**其他有影響力節點相連**的節點.

- Perfect for **undirected or weakly connected graphs** (common in mutual-follow networks) 特別適用于**無向圖或弱連通圖**（這在互相關注的社交網路中很常見）.

---
<style scoped>
section {
    font-size: 25px;
}
</style>
- Formula:

$\mathbf{x} = \alpha A \mathbf{x} + \beta \mathbf{1}$ 

Solved as: 

$\mathbf{x} = \beta (I - \alpha A)^{-1} \mathbf{1}$ 

Typical values: $\alpha = 0.05–0.2\beta = 1$

In practice: **High Katz = influential + connected to influential people** → ideal for finding real KOLs 在實際應用中：**Katz 中心性高 = 自身有影響力 + 且與有影響力的人相連** → 非常適合用於識別真正意義上的關鍵意見領袖（KOL）.

---

###### Centrality Analysis
<style scoped>
  pre {
    max-height: 400px; /* Adjust height as needed */
    overflow-y: auto;
    font-size: 2.8rem; /* Optional: adjust font size to fit more lines */
  }
</style>
```py
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import defaultdict

# --------------------------------------------------------------
# 1. Generate synthetic social network (100 users, 4 topics)
# --------------------------------------------------------------
n_users = 100
n_topics = 4
users_per_topic = n_users // n_topics

topic_names = ["Sports", "Politics", "AI_Research", "Gaming"]
true_labels = [i // users_per_topic for i in range(n_users)]

p_within = 0.25
p_between = 0.02

G = nx.Graph()
G.add_nodes_from(range(n_users))

for i in range(n_users):
    for j in range(i+1, n_users):
        if true_labels[i] == true_labels[j]:
            if random.random() < p_within:
                G.add_edge(i, j)
        else:
            if random.random() < p_between:
                G.add_edge(i, j)

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
```

---
<style scoped>
  pre {
    max-height: 400px; /* Adjust height as needed */
    overflow-y: auto;
    font-size: 2.8rem; /* Optional: adjust font size to fit more lines */
  }
</style>
```py
# --------------------------------------------------------------
# 2. Compute ALL centrality measures
# --------------------------------------------------------------

# 1. Degree Centrality
degree_centrality = nx.degree_centrality(G)

# 2. Betweenness Centrality (approximated for speed)
betweenness_centrality = nx.betweenness_centrality(G, k=100, seed=42)

# 3. Closeness Centrality
closeness_centrality = nx.closeness_centrality(G)

# 4. Eigenvector Centrality
eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)

# 5. PageRank
pagerank = nx.pagerank(G, alpha=0.85)

# 6. NEW: Katz Centrality
# Choose α < 1/λ_max where λ_max is the largest eigenvalue of adjacency matrix
try:
    # Safest: use NetworkX built-in (automatically chooses α)
    katz_centrality = nx.katz_centrality(G, alpha=0.1, beta=1.0, max_iter=1000, tol=1e-6)
except:
    # Fallback: conservative α
    katz_centrality = nx.katz_centrality(G, alpha=0.05, beta=1.0)
```

---
<style scoped>
  pre {
    max-height: 400px; /* Adjust height as needed */
    overflow-y: auto;
    font-size: 2.8rem; /* Optional: adjust font size to fit more lines */
  }
</style>

```py
# --------------------------------------------------------------
# 3. Get top 5 influencers for each measure
# --------------------------------------------------------------
def get_top_kols(centrality_dict, k=5):
    return sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:k]

top_katz = get_top_kols(katz_centrality)

centralities = {
    "Degree": get_top_kols(degree_centrality),
    "Betweenness": get_top_kols(betweenness_centrality),
    "Closeness": get_top_kols(closeness_centrality),
    "Eigenvector": get_top_kols(eigenvector_centrality),
    "PageRank": get_top_kols(pagerank),
    "Katz": top_katz
}

# --------------------------------------------------------------
# 4. Print results with topic labels
# --------------------------------------------------------------
print("\n" + "="*80)
print("TOP 5 INFLUENCERS (KOLs) BY CENTRALITY MEASURE")
print("="*80)

for name, tops in centralities.items():
    print(f"\n{name:12} Centrality → Top 5 Users:")
    for rank, (user, score) in enumerate(tops, 1):
        topic = topic_names[true_labels[user]]
        print(f"  #{rank} User {user:2d} ({topic:12}) → Score: {score:.4f}")

# --------------------------------------------------------------
# 5. Visualize: Node size by Katz Centrality
# --------------------------------------------------------------
plt.figure(figsize=(12, 9))
pos = nx.spring_layout(G, k=0.18, iterations=60, seed=42)

# Node sizes scaled by Katz centrality
katz_values = [katz_centrality[node] for node in G.nodes()]
node_sizes = [2000 * v + 100 for v in katz_values]  # scale up

# Color by true topic
node_colors = true_labels

nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.5)
scatter = nx.draw_networkx_nodes(
    G, pos,
    node_size=node_sizes,
    node_color=node_colors,
    cmap=plt.cm.tab10,
    alpha=0.9
)

# Highlight top 3 Katz influencers with red border
top3_katz_users = [u for u, _ in top_katz[:3]]
nx.draw_networkx_nodes(
    G, pos,
    nodelist=top3_katz_users,
    node_size=800,
    node_color="red",
    edgecolors="black",
    linewidths=3,
    label="Top 3 Katz KOLs"
)

plt.title("100-User Social Network\nNode Size = Katz Centrality | Red Border = Top 3 KOLs", 
          fontsize=14, pad=20)
plt.axis("off")
plt.legend(scatterpoints=1)
plt.tight_layout()
plt.show()
```

---
<style scoped>
  pre {
    max-height: 400px; /* Adjust height as needed */
    overflow-y: auto;
    font-size: 2.8rem; /* Optional: adjust font size to fit more lines */
  }
</style>
```py
# --------------------------------------------------------------
# 6. Bonus: Correlation between Katz and other measures
# --------------------------------------------------------------
import pandas as pd

df = pd.DataFrame({
    'Degree': degree_centrality,
    'Betweenness': betweenness_centrality,
    'Closeness': closeness_centrality,
    'Eigenvector': eigenvector_centrality,
    'PageRank': pagerank,
    'Katz': katz_centrality
})

print("\nCorrelation with Katz Centrality:")
print(df.corr()['Katz'].round(4).sort_values(ascending=False))
```



---

## Bridge (Cross-Ideology) Detection
<style scoped>
section {
    font-size: 26px;
}
</style>
In social media network analysis, particularly in studies of political polarization, a **"bridge"** or **"cross-ideology bridge"** refers to a user (node) who connects otherwise separated ideological communities (e.g., liberal vs. conservative users on Twitter/X). 在社交媒體網路分析中，尤其是在政治極化研究中，“**橋樑**”（bridge）或“**跨意識形態橋樑**”（cross-ideology bridge）指的是那些連接原本彼此隔離的意識形態群體（例如 Twitter/X 上自由派與保守派用戶）的用戶（節點）。

These bridges are rare but important because they are the few accounts through which information, memes, or influence can flow between opposing sides.  這類橋樑十分稀少，卻至關重要——因為它們是資訊、迷因（meme）或影響力得以在對立陣營之間流動的少數通道。

---
<style scoped>
section {
    font-size: 25px;
}
</style>
Nodes with **high betweenness centrality** lie on many shortest paths between other nodes. 具有**高仲介中心性**（high betweenness centrality）的節點，位於大量其他節點對之間的最短路徑上。

In a polarized network, the graph naturally splits into two (or more) densely connected ideological clusters with very few connections between them. 在一個高度極化的網路中，圖結構自然分裂為兩個（或多個）內部連接緊密的意識形態簇群，而簇群之間的連接極少。

The `few existing connections` between clusters make certain nodes lie on almost all shortest paths that cross from one side to the other → those nodes get extremely high betweenness scores. 正是這些**為數不多的跨簇連接**，使得某些節點幾乎出現在所有從一側到另一側的最短路徑上，從而獲得極高的仲介中心性得分。



---

###### Why Betweenness Works So Well for Cross-Ideology Detection **為何仲介中心性在識別跨意識形態橋樑時如此有效？**
<style scoped>
section {
    font-size: 25px;
}
</style>
- In highly polarized networks, the graph is almost disconnected between ideologies 在高度極化的網路中，不同意識形態群體之間的連接近乎斷開. 
- Almost any path from a left-leaning user to a right-leaning user must go through one of the rare cross-ideology edges 幾乎任何從左傾使用者到右傾使用者的路徑，都必須經過那幾條罕見的跨意識形態邊.


- The users at the ends of those rare cross-ideology edges (or users who relay them) end up with disproportionately high betweenness centrality 這些跨意識形態邊兩端的用戶（或轉發/中繼這些資訊的使用者），因此佔據了大量跨群體最短路徑的關鍵位置，導致其**仲介中心性顯著高於其他節點**.
- Therefore, the top-ranked nodes by betweenness centrality are almost always genuine cross-ideology bridges 因此，按仲介中心性排序靠前的節點，幾乎總是真正的跨意識形態橋樑。.

---
<style scoped>
  pre {
    max-height: 400px; /* Adjust height as needed */
    overflow-y: auto;
    font-size: 2.8rem; /* Optional: adjust font size to fit more lines */
  }
</style>
```py
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# ========================================
# 1. Create a polarized network with 100 users
# ========================================
n_left = 50   # Left community
n_right = 50  # Right community
n = n_left + n_right

G = nx.Graph()

# Add nodes with ideology attribute
for i in range(n_left):
    G.add_node(f"L{i}", ideology="Left")
for i in range(n_right):
    G.add_node(f"R{i}", ideology="Right")

# ========================================
# 2. Add dense connections WITHIN each side (homophily)
# ========================================
# Left community: high connection probability
for i in range(n_left):
    for j in range(i+1, n_left):
        if random.random() < 0.25:  # dense
            G.add_edge(f"L{i}", f"L{j}")

# Right community: high connection probability
for i in range(n_right):
    for j in range(i+1, n_right):
        if random.random() < 0.25:
            G.add_edge(f"R{i}", f"R{j}")

# ========================================
# 3. Add ONLY 4 cross-ideology bridges (the interesting part!)
# ========================================
bridges = [
    ("L5",  "R8"),   # a journalist both sides follow
    ("L12", "R3"),   # a moderate politician
    ("L23", "R19"),  # local news outlet
    ("L41", "R33")   # satire account
]

for a, b in bridges:
    G.add_edge(a, b)
```

---

<style scoped>
  pre {
    max-height: 400px; /* Adjust height as needed */
    overflow-y: auto;
    font-size: 2.8rem; /* Optional: adjust font size to fit more lines */
  }
</style>

```py
# ========================================
# 4. Compute betweenness centrality
# ========================================
betweenness = nx.betweenness_centrality(G)

# Convert to sorted list
sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)

print("Top 15 nodes by betweenness centrality:")
print("-" * 50)
for node, bc in sorted_betweenness[:15]:
    ideology = G.nodes[node]['ideology']
    is_bridge = "BRIDGE" if any((node == a or node == b) for a, b in bridges) else ""
    print(f"{node:4} | Betweenness: {bc:6.4f} | {ideology:5} {is_bridge}")
```
---
<style scoped>
  pre {
    max-height: 400px; /* Adjust height as needed */
    overflow-y: auto;
    font-size: 2.8rem; /* Optional: adjust font size to fit more lines */
  }
</style>

```py
# ========================================
# 5. Visualize the network
# ========================================
plt.figure(figsize=(12, 8))

pos = nx.spring_layout(G, seed=42)

# Color by ideology
node_colors = ['#ff4444' if G.nodes[n]['ideology'] == 'Left' else '#4444ff' for n in G.nodes()]

# Make bridge nodes larger and yellow
node_sizes = []
for node in G.nodes():
    if any((node == a or node == b) for a, b in bridges):
        node_sizes.append(400)
    else:
        node_sizes.append(80)

nx.draw(G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        with_labels=False,
        edge_color='gray',
        alpha=0.7)

# Highlight bridges in yellow
bridge_nodes = set()
for a, b in bridges:
    bridge_nodes.add(a)
    bridge_nodes.add(b)

nx.draw_networkx_nodes(G, pos,
                       nodelist=bridge_nodes,
                       node_color='yellow',
                       node_size=500,
                       edgecolors='black',
                       linewidths=2)

plt.title("Polarized Network with 4 Cross-Ideology Bridges (yellow nodes)\n"
          "Betweenness centrality perfectly detects them!", fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.show()
```



---

## Churn Prevention
<style scoped>
section {
    font-size: 25px;
}
</style>
**Churn prevention** is a critical use case where graph theory shines—especially by identifying isolated or disengaged customers (孤立狀態或參與度下降的客戶) who are at high risk of leaving. **客戶流失預防**是圖論大顯身手的關鍵應用場景之一——尤其體現在識別那些處於**孤立狀態**或**參與度下降**的客戶，他們往往具有較高的流失風險。

In a social or interaction-based product, **low connectivity** often signals **disengagement**, **lack of network effects**, or **missing value**—all strong churn predictors. 在依賴社交互動或使用者間連接的產品中, **低連接度**（low connectivity）通常預示著: 用戶參與度降低; 未能從網路效應中獲益; 未感知到產品核心價值.

這些因素都是預測客戶流失的強有力信號。通過將用戶建模為圖中的節點、互動行為（如消息、協作、邀請、點贊等）建模為邊，企業可利用圖分析技術（如度中心性、連通分量、社區檢測等）主動識別高風險用戶，並及時干預，從而有效降低流失率。

---
<style scoped>
section {
    font-size: 25px;
}
</style>
We can compute metrics like **degree**, **ego network size**, and **clustering coefficient** to flag "at-risk" (churn-prone) customers. These metrics capture engagement levels and social "stickiness":

- **Degree = 0**: Isolated nodes with no connections indicate completely disengaged customers, highly likely to churn due to lack of activity or ties. **節點度**（Degree）：客戶直接連接的其他節點數量（如好友、互動物件、購買品類等）。低度  可能意味著參與度低或社交孤立。

- **Ego network size = 1 or 2**: The ego network is the subgraph consisting of a node (the "ego") and its direct neighbors. Size 1 means isolated (degree 0); size 2 means only one connection (degree 1). These small egos suggest minimal integration into the network, making churn easier without social repercussions (based on homophily—similar users tend to behave alike). **自我中心網路規模**（Ego-network Size）：以客戶為中心、在特定跳數內（通常為1跳）可達的所有節點數量自我中心网络 = 用户自身 + 其直接邻居。規模過小可能反映整體網路嵌入度不足。

---
<style scoped>
section {
    font-size: 26px;
}
</style>
- **Low clustering coefficient**: This measures how densely a node's neighbors are connected to each other (fraction of possible triangles that exist). Low values (e.g., < 0.2) indicate sparse local communities, implying weak relational bonds and higher churn propensity, as the customer isn't embedded in a tight-knit group. **聚類係數**（Clustering Coefficient）：衡量客戶鄰居之間彼此連接的緊密程度。低聚類係數可能表明其社交圈鬆散或缺乏社群歸屬。
  - **高聚類係數**：使用者所處的局部網路緊密（例如，所在團隊成員之間頻繁協作），說明其深度嵌入某個活躍子群，流失風險較低。
  - **低/零聚類係數**：即使用戶有多個連接，但這些連接彼此不關聯（如隨機使用多個孤立功能），可能反映淺層參與，仍存在流失隱患。

這些指標共同刻畫了客戶在整體網路中的`嵌入性（embeddedness）`——嵌入性越低，越可能處於孤立狀態，流失風險越高。

---
<style scoped>
section {
    font-size: 25px;
}
</style>
Degree = social integration**, **Isolated nodes = abandoned users**, **Weak ties (degree=1) = one foot out the door.

- **社交融入程度**（Social Integration）：用戶在平臺或社區中與其他使用者或實體的連接廣度，通常用 度（degree）衡量。連接越多，粘性越強。

- **孤立節點**（Isolated Nodes）：度為0的節點，意味著該用戶沒有任何交互、關注、購買或其他行為關聯，極可能已流失或從未真正啟動。

- **弱連接** **/ “一隻腳出門”**（Weak Ties, degree=1）：僅有單次或單一類型的互動（如只關注一個KOL、只買過一次商品），缺乏多元連接，忠誠度低，極易流失。

---
<style scoped>
  pre {
    max-height: 400px; /* Adjust height as needed */
    overflow-y: auto;
    font-size: 2.8rem; /* Optional: adjust font size to fit more lines */
  }
</style>
```py
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# 1. Simulate Customer Interaction Data
# ----------------------------
# Example: platform where users collaborate (message, share files, mention)
# Format: (user, interacts_with)
interactions = [
    ("user1", "user2"),
    ("user1", "user3"),
    ("user2", "user3"),
    ("user4", "user5"),
    ("user5", "user4"),
    # Isolated or near-isolated users (churn risk)
    ("user6", "user7"),  # only one weak link
    ("user8", None),     # completely isolated
    ("user9", None),
    ("user10", "user11"),
    ("user11", "user10"),
    ("user12", None),    # silent user
]

# Build an undirected graph of user interactions
G = nx.Graph()

# Add all users
all_users = set()
for u, v in interactions: # 取出senter, receiver
    all_users.add(u)
    if v is not None:
        all_users.add(v)

G.add_nodes_from(all_users)

# Add edges where interaction exists
for u, v in interactions:
    if v is not None:
        G.add_edge(u, v)

print(f"Total users: {G.number_of_nodes()}")
print(f"Connected users: {len([n for n in G.nodes() if G.degree(n) > 0])}")
print(f"Isolated users: {len([n for n in G.nodes() if G.degree(n) == 0])}")
```

---
<style scoped>
  pre {
    max-height: 400px; /* Adjust height as needed */
    overflow-y: auto;
    font-size: 2.8rem; /* Optional: adjust font size to fit more lines */
  }
</style>

```py
# ----------------------------
# 2. Identify At-Risk (Churn-Prone) Customers
# ----------------------------
churn_risk = []

for user in G.nodes():
    degree = G.degree(user)
    # Ego network = user + direct neighbors
    ego = nx.ego_graph(G, user, radius=1) # <----
    
    ego_size = ego.number_of_nodes()  # 1 = isolated
    
    # Optional: clustering coefficient (how connected are their friends?)
    # Not defined for degree < 2
    clustering = nx.clustering(G, user) if degree >= 2 else 0.0
    
    # Heuristic for churn risk:
    # - Degree == 0 → completely isolated
    # - Degree == 1 and ego_size == 2 → only one weak tie
    if degree == 0 or (degree == 1 and ego_size == 2):
        churn_risk.append({
            "user": user,
            "degree": degree,
            "ego_size": ego_size,
            "clustering": clustering
        })

print(f"\n {len(churn_risk)} users flagged for high churn risk:")
for r in churn_risk:
    print(f"  {r['user']}: degree={r['degree']}, ego_size={r['ego_size']}")
```

---
<style scoped>
  pre {
    max-height: 400px; /* Adjust height as needed */
    overflow-y: auto;
    font-size: 2.8rem; /* Optional: adjust font size to fit more lines */
  }
</style>

```py
# ----------------------------
# 3. Prioritize for Retention Campaigns
# ----------------------------
# Offer onboarding help, feature tutorial, or referral bonus
retention_actions = {
    "degree=0": "Send re-engagement email + product walkthrough",
    "degree=1": "Suggest connections or team members to invite"
}

print("\n Suggested retention actions:")
for r in churn_risk:
    if r["degree"] == 0:
        action = retention_actions["degree=0"]
    else:
        action = retention_actions["degree=1"]
    print(f"  {r['user']} → {action}")

# ----------------------------
# 4. Visualize Network (Highlight At-Risk Users)
# ----------------------------
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)

# Color nodes: green = connected, red = at-risk
node_colors = []
node_sizes = []
for node in G.nodes():
    if node in [r["user"] for r in churn_risk]:
        node_colors.append("red")
        node_sizes.append(600)
    else:
        node_colors.append("lightgreen")
        node_sizes.append(400)

nx.draw(
    G, pos,
    node_color=node_colors,
    node_size=node_sizes,
    with_labels=True,
    font_size=10,
    edge_color="gray"
)

plt.title("Customer Interaction Graph\n Red = High Churn Risk (Isolated)")
plt.tight_layout()
plt.show()
```



----

##  Application in Marketing
<style scoped>
section {
    font-size: 26px;
}
</style>
`Influencer marketing` leverages social networks to promote products/ideas through key individuals who can amplify messages. 網紅行銷（Influencer marketing）借助社交網路，通過 關鍵個體 來推廣產品或理念，這些個體能夠放大傳播資訊.  

Graph theory models this as nodes (users) and edges (interactions/referrals), identifying influencers via metrics like `high-degree (broad reach)`, `high betweenness (information brokers)`, and `echo chambers (amplified niches)`. 圖論將這一過程建模為節點（用戶）和邊（互動/推薦），並通過多種指標識別關鍵影響者：**高連接度**（high-degree，代表廣泛觸達）、**高介數中心性**（high betweenness，作為資訊仲介連接不同群體），以及**回音室效應**（echo chambers，指資訊在封閉小圈子內被不斷強化的傳播現象）。

----
<style scoped>
section {
    font-size: 25px;
}
</style>
**High-degree ≠ high influence.** A user with 10K followers in one echo chamber spreads your message only *within* that bubble. 一個在單一回音室（echo chamber）中擁有 1 萬粉絲的用戶，只能將你的資訊**局限在該圈子內部傳播**

A user with **high betweenness** connects **multiple communities**—they **diffuse** your message. **高連接度** **≠** **高影響力**。 ；而一個**介數中心性**（betweenness）高的用戶，則連接著**多個不同社群**，能夠真正**擴散**（diffuse）你的資訊，實現跨圈層傳播。

- **Echo Chamber**（回音室 ）：Densely connected subgraph with high internal clustering but low external ties → opinions "echo" inside. 指用戶僅與觀點相似的人互動，資訊在封閉群體內反復迴圈，缺乏外部滲透。
  - **High-degree**（高連接度）：Has many followers and frequent interactions, but if these connections are confined within a single community, the reach of information remains limited. 粉絲多、互動多，但若都在同一社群內，傳播廣度有限。
  - **High Betweenness**（高介數中心性）：Acts as a "bridge" node between multiple communities. Even with fewer followers, such users can effectively relay information from one group to another, thereby **unlocking network-wide diffusion**. 位於多個社群之間的“橋樑”節點，即使粉絲不多，也能將資訊從一個群體傳遞到另一個群體，**撬動全域傳播**。

---
<style scoped>
section {
    font-size: 25px;
}
</style>
Thus, **identify bridge users**: Use graph analysis to find ordinary users with high betweenness centrality—such as employees who span departments or individuals active across diverse interest-based communities—as they may be far more effective spreaders than highly connected but insular users. so **識別橋樑用戶**：通過圖分析找出 介數中心性高的普通用戶（如跨部門員工、多興趣社群參與者），他們可能是更高效的傳播節點。

- **策略調整**：In seeding campaigns or word-of-mouth marketing, prioritize users who **connect different communities**. True influence isn’t about how many people you know—it’s about how many **mutually disconnected worlds** you link together. 在種子用戶投放或口碑行銷中，優先選擇**連接不同社區的用戶**. 真正的影響力，不在於你認識多少人，而在於你連接了多少個**彼此不認識的世界**。

---
<style scoped>
  pre {
    max-height: 400px; /* Adjust height as needed */
    overflow-y: auto;
    font-size: 2.8rem; /* Optional: adjust font size to fit more lines */
  }
</style>
```py
import networkx as nx
import matplotlib.pyplot as plt

# ----------------------------
# 1. Simulate a Referral/Interaction Network
# ----------------------------
# Format: (referrer, referred) — or (user1, user2) for mutual interaction
edges = [
    # Community A: Tech enthusiasts
    ("Alice", "Bob"),
    ("Bob", "Charlie"),
    ("Alice", "Charlie"),
    ("Charlie", "Diana"),
    
    # Community B: Fitness group
    ("Eve", "Frank"),
    ("Frank", "Grace"),
    ("Eve", "Grace"),
    
    # Community C: Sustainability advocates
    ("Helen", "Ivy"),
    ("Ivy", "Jack"),
    ("Helen", "Jack"),
    
    #  BRIDGE NODES (potential influencers)
    ("Diana", "Eve"),   # connects Tech ↔ Fitness
    ("Grace", "Helen"), # connects Fitness ↔ Sustainability
    ("Jack", "Alice"),  # connects Sustainability ↔ Tech → forms a cycle
]

# Build an undirected graph (referrals are mutual in influence)
G = nx.Graph()
G.add_edges_from(edges) # edge list

print(f"Network: {G.number_of_nodes()} users, {G.number_of_edges()} connections")
```

---
<style scoped>
  pre {
    max-height: 400px; /* Adjust height as needed */
    overflow-y: auto;
    font-size: 2.8rem; /* Optional: adjust font size to fit more lines */
  }
</style>

```py
# ----------------------------
# 2. Compute Betweenness Centrality
# ----------------------------
betweenness = nx.betweenness_centrality(G)

# Sort by betweenness (descending)
influencers = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)

print("\n Top Influencers by Betweenness Centrality:")
print("User\t\tBetweenness\tRole")
print("-" * 45)
for user, score in influencers[:5]:
    role = "Bridge" if score > 0.2 else "Local"
    print(f"{user:<12}\t{score:.3f}\t\t{role}")

# ----------------------------
# 3. Why Betweenness > Degree?
# ----------------------------
degree = dict(G.degree())
print("\n Degree vs. Betweenness (Top 5 by degree):")
print("User\tDegree\tBetweenness")
print("-" * 30)
for user in sorted(degree, key=degree.get, reverse=True)[:5]:
    print(f"{user:<12}\t{degree[user]}\t{betweenness[user]:.3f}")

# → Notice: high-degree users may have LOW betweenness!
```

---
<style scoped>
  pre {
    max-height: 400px; /* Adjust height as needed */
    overflow-y: auto;
    font-size: 2.8rem; /* Optional: adjust font size to fit more lines */
  }
</style>

```py
# ----------------------------
# 4. Visualize: Highlight Bridge Influencers
# ----------------------------
plt.figure(figsize=(12, 8))

# Get the current axes
ax = plt.gca()

# Layout
pos = nx.spring_layout(G, seed=42)

# Color nodes by betweenness (red = high)
node_colors = [betweenness[node] for node in G.nodes()]
node_sizes = [3000 * betweenness[node] + 300 for node in G.nodes()]

# Draw the graph
nx.draw(
    G, pos,
    node_color=node_colors,
    cmap=plt.cm.plasma,
    node_size=node_sizes,
    with_labels=True,
    font_size=10,
    edge_color="gray",
    width=1.5,
    ax=ax  # Explicitly pass ax
)

# Add colorbar using the same axes
sm = plt.cm.ScalarMappable(
    cmap=plt.cm.plasma,
    norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors))
)
sm.set_array([])  # Required for older matplotlib versions
plt.colorbar(sm, ax=ax, label="Betweenness Centrality")  # <-- pass ax here

plt.title("Referral Network: Node Size & Color = Betweenness Centrality\n Red/Large = Key Influencers (Bridge Communities)")
plt.tight_layout()
plt.show()

# ----------------------------
# 5. Marketing Action: Target Bridge Influencers
# ----------------------------
top_influencer = influencers[0][0]
print(f"\n Recommendation: Partner with '{top_influencer}' for campaign.")
print("Why? They connect multiple communities—maximizing spread and diversity of reach.")
```





---

## Application in Finance

**Portfolio Risk Using Graph Theory**
<style scoped>
section {
    font-size: 25px;
}
</style>
Financial assets (stocks, bonds, derivatives) exhibit `interdependencies` driven by market dynamics, sectors, or macroeconomic factors. Graph theory helps model and analyze these dependencies.  


###### Minimum Spanning Trees (MSTs)
It is a powerful tool in finance for extracting a **robust and interpretable skeletal structure** from complex correlation networks, helping to uncover intrinsic market linkages, identify key nodes, and support informed decision-making. **最小生成樹**在金融中是一種強大的工具，用於從複雜的相關性網路中提取**穩健、可解釋的骨架結構**，説明理解市場內在聯繫、識別關鍵節點，並支持決策制定。

---
<style scoped>
section {
    font-size: 25px;
}
</style>
MSTs achieve this by revealing the **core dependency structure and interrelationships** among a large set of financial assets—such as stocks, indices, or currencies—in the most concise and non-redundant way possible.**最小生成樹**（Minimum Spanning Trees, MSTs），是因為它能以最簡潔、無冗餘的方式揭示大量金融資產（如股票、指數或貨幣）之間的**核心依賴結構和關聯關係**。

- **Noise Reduction and Network Simplification**：Financial market data often contain substantial noise and redundant correlations. The MST constructs an acyclic, connected subgraph by retaining only the minimal set of edges needed to connect all nodes (e.g., stocks), with the total edge weight minimized (typically based on distance metrics derived from negative correlations or dissimilarity measures). This filters out weaker or spurious relationships and highlights the most significant co-movements.金融市場資料通常包含大量雜訊和冗餘相關性。MST 通過保留連接所有節點（如股票）所需的最少邊（且總權重最小，通常基於負相關性或距離度量），構建一個無環、連通的子圖，從而過濾掉次要關聯，突出最重要的聯動關係。


---
<style scoped>
section {
    font-size: 25px;
}
</style>
- **Identifying Systemic Risk and Sector Structure**：By analyzing the MST’s topology—such as hub nodes and branch lengths—one can detect which assets serve as “hubs” (potentially playing a critical role in systemic risk transmission) and how the market naturally clusters into sectors or groups通過觀察 MST 的拓撲結構（如中心節點、分支長度），可以發現哪些資產處於“樞紐”位置（可能對系統性風險傳導起關鍵作用），以及市場如何分組。

- **Monitoring Market Dynamics**：Over time, structural changes in the MST—such as shifts in central hubs or sector reconfigurations—can reflect evolving market conditions (e.g., a sharp increase in cross-asset correlations during a financial crisis), providing early warning signals for risk management.隨著時間推移，MST 的結構變化（如中心節點轉移、板塊重組）可反映市場狀態的演變（如金融危機期間不同資產間相關性驟增），為風險管理提供早期信號。

---

###### From Raw Financial Data to MST
<style scoped>
section {
    font-size: 26px;
}
</style>
Returns --> correlation matrix --> distance --> weight graph --> MST algorithm

1. **Compute Returns**: Convert prices to **log returns** (or simple returns). Returns are stationary and better reflect risk/dependence than raw prices
2. Compute the **Pearson correlation coefficient** between every pair of assets.
3. **Transform Correlation into Distance**: Correlation is **not a metric** (doesn’t satisfy triangle inequality), so we convert it to a **distance measure** by the **ultrametric distance** (Mantegna, 1999): $ d_{ij}=\sqrt{2(1-\rho_{ij})}$ , where $\rho=1, d_{ij}=0;\rho=-1,d_{ij}=2$.

4. **Build a Complete Weighted Graph**: Treat each asset as a **node**. Connect every pair of nodes with an **edge weighted by** $d_{ij}$.
5. Apply a standard MST algorithm (e.g., **Kruskal’s** or **Prim’s**) to the weighted graph. 

---

**Systemic Risk Identification** **系統性風險識別**:
<style scoped>
section {
    font-size: 25px;
}
</style>
- Highly connected or central assets (high **degree** or **eigenvector centrality**) can act as `risk conduits`. **高度連接或處於核心地位的資產**（具有較高的**度中心性**或**特徵向量中心性**）可能成為風險傳導的管道 

- **Clustering** identifies sectors or asset classes that move together, exposing concentration risk. **聚類分析**可識別出價格走勢高度一致的行業或資產類別，從而揭示**集中度風險**（即風險過度集中於某一類資產或部門）。

---
<style scoped>
section {
    font-size: 25px;
}
</style>  

**Stress Testing via Graph Perturbation**: **通過圖擾動進行壓力測試**

- Simulate shocks (e.g., asset default) and propagate effects through the graph using diffusion models or cascade algorithms. 模擬衝擊事件（例如某項資產違約），並利用**擴散模型**（diffusion models）或**級聯演算法**（cascade algorithms）在圖中傳播其影響，以評估整個系統在極端情景下的脆弱性與風險傳染路徑。

**Diversification Optimization**: **分散化優化**

- Select assets from weakly connected graph components to minimize co-movement. 從圖中**連接較弱的子圖**（或弱連通成分）中選取資產，以降低資產間的**聯動性**（co-movement），從而增強投資組合的風險分散效果。

---
<style scoped>
  pre {
    max-height: 400px; /* Adjust height as needed */
    overflow-y: auto;
    font-size: 2.8rem; /* Optional: adjust font size to fit more lines */
  }
</style>

```py
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from sklearn.datasets import make_spd_matrix

# ----------------------------
# 1. Simulate Stock Returns
# ----------------------------
np.random.seed(42)
n_assets = 10
n_days = 250

# Generate a realistic (positive semi-definite) covariance matrix
cov_matrix = make_spd_matrix(n_assets, random_state=42)
# Simulate daily returns from multivariate normal
returns = np.random.multivariate_normal(mean=np.zeros(n_assets), cov=cov_matrix, size=n_days)

# Create a DataFrame with asset labels
assets = [f"Asset_{i}" for i in range(1, n_assets + 1)]
returns_df = pd.DataFrame(returns, columns=assets)
```

---
<style scoped>
  pre {
    max-height: 400px; /* Adjust height as needed */
    overflow-y: auto;
    font-size: 2.8rem; /* Optional: adjust font size to fit more lines */
  }
</style>
```py
# ----------------------------
# 2. Compute Correlation Matrix
# ----------------------------
corr = returns_df.corr()
print("Correlation matrix (top-left 5x5):")
print(corr.iloc[:5, :5].round(2))

# ----------------------------
# 3. Build Correlation Network
# ----------------------------
# Convert correlation to distance: d = sqrt(0.5 * (1 - corr))
dist_matrix = np.sqrt(0.5 * (1 - corr))

# Create a complete graph with distance as edge weight
G = nx.Graph()
for i in range(n_assets):
    for j in range(i + 1, n_assets):
        asset_i = assets[i]
        asset_j = assets[j]
        distance = dist_matrix.iloc[i, j]
        G.add_edge(asset_i, asset_j, weight=distance)

        # ----------------------------
# 4. Extract Minimum Spanning Tree (MST)
# ----------------------------
mst = nx.minimum_spanning_tree(G, weight='weight')
```

---

**Why Look for High-Degree or High-Betweenness Nodes in an MST?** 為什麼在 MST 中找高度數或高介數節點？

**High Degree Centrality**
<style scoped>
section {
    font-size: 26px;
}
</style>
- Indicates that the asset is **directly connected to many other assets** in the MST.  表示該資產在 MST 中直接連接了**很多其他資產**。
- Suggests it serves as a **convergence point for multiple relationships**—if this asset experiences a shock (e.g., a sharp price drop or default), it could **immediately affect numerous other assets**. 說明它是**多個關聯關係的交匯點**，一旦該資產發生衝擊（如價格暴跌、違約），可能**直接波及眾多其他資產**。
- Retaining high connectivity even in the simplified MST implies its **centrality is structural, not due to noise**, underscoring its genuine systemic importance在精簡後的網路中仍擁有高度連接，說明其**核心地位不是由雜訊導致的**，而是結構性的。

---

**High Betweenness Centrality**
<style scoped>
section {
    font-size: 25px;
}
</style>
- Means the asset lies on a **large number of shortest paths** (which, in an MST, are the *only* paths between node pairs). 表示該資產位於**大量最短路徑**（在 MST 中是唯一路徑）。
- Even if it has few direct neighbors, information or risk transmission between many asset pairs must pass through it.即使它連接的鄰居不多，但很多資產之間的資訊/風險傳導必須經過它。
- Such nodes act as **“bridges” or “bottlenecks” in risk propagation**—if they fail or become disrupted, they can block, reroute, or amplify systemic risk flows, potentially triggering widespread cascading effects這類節點是風險傳導的“橋樑”或“瓶頸”**—— 一旦失效，可能阻斷或扭曲整個系統的風險傳播路徑，甚至引發更大範圍的連鎖反應。

---
<style scoped>
  pre {
    max-height: 400px; /* Adjust height as needed */
    overflow-y: auto;
    font-size: 2.8rem; /* Optional: adjust font size to fit more lines */
  }
</style>
```py
# ----------------------------
# 5. Analyze Risk: Centrality in MST
# ----------------------------
# In MST, high degree or betweenness = systemic risk hub
degree_centrality = nx.degree_centrality(mst)
betweenness = nx.betweenness_centrality(mst)

# Rank assets by betweenness (most influential in risk propagation)
risk_ranking = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
print("\nTop risk-conduit assets (by betweenness centrality in MST):")
for asset, score in risk_ranking[:3]:
    print(f"  {asset}: {score:.3f}")
    
# ----------------------------
# 6. Visualize MST
# ----------------------------
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(mst, seed=42)

# Node size proportional to betweenness
node_sizes = [5000 * betweenness[node] + 300 for node in mst.nodes()]

nx.draw(mst, pos,
        with_labels=True,
        node_size=node_sizes,
        node_color='lightgreen',
        font_size=9,
        edge_color='gray')

plt.title("Portfolio Risk Network: Minimum Spanning Tree\n(Node size ∝ risk influence)")
plt.tight_layout()
plt.show()    
```



----

## Supply Chain Finance & Counterparty Risk
<style scoped>
section {
    font-size: 25px;
}
</style>
 In today’s globalized supply chains, firms are not just independent business entities—they are **nodes within a complex supplier–customer network**. 在現代全球化供應鏈中，企業不僅是獨立的經營主體，更是複雜供應網路中的一個節點。 When a critical supplier or customer suffers an operational disruption (e.g., bankruptcy, natural disaster, or geopolitical conflict), risk propagates rapidly through transactional relationships, potentially triggering cascading production halts, contract defaults, and financing difficulties. This is precisely the essence of **counterparty risk**. 一旦關鍵供應商或客戶發生經營中斷（如破產、自然災害、地緣衝突），風險會通過**交易關係**迅速傳導，造成**連鎖停工、訂單違約、融資困難**等後果——這正是**交易對手風險**（Counterparty Risk）的核心體現。

---
<style scoped>
section {
    font-size: 25px;
}
</style>
To enhance supply chain resilience, we can model the supply chain as a **directed supplier–customer network** and apply graph-theoretic tools to identify vulnerabilities, assess shock impacts, and generate optimization recommendations. 為提升供應鏈韌性，可將供應鏈建模為**供應商–客戶網路**，並借助圖論工具識別脆弱點、評估衝擊影響，並提出優化建議。



- Build a supplier–customer network, **Nodes** = firms (suppliers, manufacturers, distributors, etc.). **Directed edges**: An edge A→B indicates that firm A supplies goods to firm B (i.e., B depends on A).節點（Node = 企業（供應商、製造商、分銷商等）, 有向邊（Edge） A → B = 企業 A 向企業 B 供貨（B 依賴 A）.  **Edge weights (optional)**: transaction value, share of supply, delivery frequency, etc.邊權重（可選）= 交易金額、供應占比、交付頻率等.

---
<style scoped>
section {
    font-size: 25px;
}
</style>
- In **supply chain finance**, if firm B is a financed client and firm A is its key supplier, then A’s stability directly affects B’s ability to fulfill obligations—thereby influencing the lender’s credit risk exposure to B. 在供應鏈金融中，若 B 是融資客戶，A 是其關鍵供應商，則 A 的穩定性直接影響 B 的履約能力，進而影響金融機構對 B 的授信風險。

- Compute **centrality metrics** (PageRank, betweenness) to find critical firms,
  - **PageRank**：Measures a firm’s overall “influence” or “level of dependency” in the network 衡量企業在整個網路中的“影響力”或“被依賴程度” → *High PageRank firms* are core suppliers relied upon by many downstream customers. 高 PageRank 企業 = 被眾多下游客戶依賴的**核心供應商**
  - **Betweenness Centrality**: Measures whether a firm lies on critical paths between other firms 衡量企業是否處於**關鍵路徑**上 → *High-betweenness firms* act as “choke points”—if disrupted, they can sever multiple supply chains simultaneously 高介數企業 = **“**咽喉節點，一旦中斷將切斷多條供應鏈

---
<style scoped>
section {
    font-size: 25px;
}
</style>
- Simulate **supplier disruption impact**,
  - Assume a key supplier suddenly halts production (e.g., due to fire or sanctions). 假設某關鍵供應商突然停產（如因火災、制裁）；Assess the immediate production disruption risk for its direct customers. 評估其**直接客戶**的生產中斷風險；Propagate the shock further to estimate **second- and third-tier effects** (e.g., customers cannot deliver → their customers are also affected). 進一步類比**二級、三級影響**（客戶無法交貨 → 客戶的客戶也受影響）；Quantify the resulting **aggregate output loss** or **increase in default probability** across the entire network量化整體網路的**產出損失**或**違約概率上升**。

----
<style scoped>
  pre {
    max-height: 400px; /* Adjust height as needed */
    overflow-y: auto;
    font-size: 2.8rem; /* Optional: adjust font size to fit more lines */
  }
</style>

```py
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

# ----------------------------
# 1. Build Dense Supply Network (30 firms)
# ----------------------------
np.random.seed(42)
n_firms = 30
firm_names = [f"Firm_{i}" for i in range(n_firms)]

G = nx.DiGraph()
G.add_nodes_from(firm_names)

# Increase connectivity to ensure cascade
for i in range(n_firms):
    num_customers = np.random.randint(3, 6)  # Each firm supplies 3–5 others
    candidates = [f for f in firm_names if f != firm_names[i]]
    if candidates:
        customers = np.random.choice(candidates, size=min(num_customers, len(candidates)), replace=False)
        for c in customers:
            G.add_edge(firm_names[i], c)

# Assign daily production capacity (output)
firm_data = {firm: {'capacity': np.random.uniform(100, 600)} for firm in firm_names}

# Assign supply dependency shares
for customer in firm_names:
    suppliers = list(G.predecessors(customer))
    if suppliers:
        weights = np.random.dirichlet(np.ones(len(suppliers)) * 1.5)
        for i, supplier in enumerate(suppliers):
            G[supplier][customer]['supply_share'] = weights[i]

print(f"Network: {G.number_of_nodes()} firms, {G.number_of_edges()} supply links")

# ----------------------------
# 2. Compute Centrality & Choose Disruption Target
# ----------------------------
pagerank = nx.pagerank(G, weight='supply_share', alpha=0.85)
G_undir = G.to_undirected()
betweenness = nx.betweenness_centrality(G_undir, weight='supply_share')

# Pick firm with highest combined centrality
combined_score = {f: pagerank[f] + betweenness[f] for f in firm_names}
disrupted_firm = max(combined_score, key=combined_score.get)

print(f"\n Disrupting: {disrupted_firm}")
print(f"   PageRank: {pagerank[disrupted_firm]:.4f} | Betweenness: {betweenness[disrupted_firm]:.4f}")

# ----------------------------
# 3. Simulate Disruption Cascade
# ----------------------------
def simulate_disruption_cascade(G, firm_data, disrupted_firm, threshold=0.45, max_tiers=4):
    status = {f: 'operational' for f in firm_names}
    status[disrupted_firm] = 'disrupted'
    affected_by_tier = {0: {disrupted_firm}}
    newly_disrupted = {disrupted_firm}

    for tier in range(1, max_tiers + 1):
        next_disrupted = set()
        for firm in newly_disrupted:
            for customer in G.successors(firm):
                if status[customer] != 'operational':
                    continue
                total_loss = sum(
                    G[sup][customer].get('supply_share', 0)
                    for sup in G.predecessors(customer)
                    if status[sup] == 'disrupted'
                )
                if total_loss >= threshold:
                    status[customer] = 'disrupted'
                    next_disrupted.add(customer)
        if not next_disrupted:
            break
        affected_by_tier[tier] = next_disrupted
        newly_disrupted = next_disrupted

    total_output_loss = sum(firm_data[f]['capacity'] for f in firm_names if status[f] == 'disrupted')
    default_count = sum(1 for s in status.values() if s == 'disrupted')
    return affected_by_tier, total_output_loss, default_count, status

affected, total_loss, num_disrupted, final_status = simulate_disruption_cascade(
    G, firm_data, disrupted_firm, threshold=0.45, max_tiers=4
)

# ----------------------------
# 4. Print Detailed Disruption Report
# ----------------------------
print("\n" + "="*60)
print(" DISRUPTION IMPACT REPORT")
print("="*60)

# List all disrupted firms with their output loss
disrupted_list = []
for firm in firm_names:
    if final_status[firm] == 'disrupted':
        loss = firm_data[firm]['capacity']
        tier = next((t for t, firms in affected.items() if firm in firms), None)
        disrupted_list.append({
            'Firm': firm,
            'Tier': tier,
            'Daily Output Loss': loss
        })

# Create DataFrame for clean display
df = pd.DataFrame(disrupted_list)
df = df.sort_values(['Tier', 'Daily Output Loss'], ascending=[True, False])

print(df.to_string(index=False, float_format="{:.1f}".format))

print(f"\n TOTAL SYSTEM IMPACT:")
print(f"   • Disrupted firms: {num_disrupted} / {n_firms} ({100 * num_disrupted / n_firms:.1f}%)")
print(f"   • Total daily output loss: {total_loss:.1f} units")
total_capacity = sum(f['capacity'] for f in firm_data.values())
print(f"   • System-wide capacity loss: {100 * total_loss / total_capacity:.1f}%")

# ----------------------------
# 5. Visualization
# ----------------------------
pos = nx.spring_layout(G, seed=42, k=2.5, iterations=100)

# Safe max_tier to avoid division by zero
max_tier = max(1, max(affected.keys()))

tier_color_map = {}
for tier, firms in affected.items():
    color = plt.cm.plasma(tier / max_tier)
    for f in firms:
        tier_color_map[f] = color

node_colors = [tier_color_map.get(node, (0.85, 0.85, 0.85, 1.0)) for node in G.nodes()]
node_sizes = [200 + 3 * firm_data[node]['capacity'] / 10 for node in G.nodes()]

plt.figure(figsize=(14, 10))

nx.draw_networkx_edges(G, pos, edge_color='lightgray', arrows=True, arrowsize=10, width=0.8, alpha=0.7)
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.95)

# Label ALL disrupted firms
disrupted_nodes = [n for n, s in final_status.items() if s == 'disrupted']
nx.draw_networkx_labels(G, pos, labels={n: n for n in disrupted_nodes}, font_size=9, font_weight='bold')

# Legend
legend_elements = [
    Patch(facecolor=plt.cm.plasma(t / max_tier), label=f'Tier {t} (n={len(affected[t])})')
    for t in sorted(affected.keys())
]
legend_elements.append(Patch(facecolor=(0.85, 0.85, 0.85), label='Unaffected'))
plt.legend(handles=legend_elements, loc='upper right', title="Disruption Tier")

plt.title(
    f"Supply Chain Disruption: {disrupted_firm} Failed\n"
    f"{num_disrupted}/{n_firms} firms disrupted | {100 * total_loss / total_capacity:.1f}% output loss",
    fontsize=14, pad=20
)
plt.axis('off')
plt.tight_layout()
plt.show()
```

---
<style scoped>
  pre {
    max-height: 400px; /* Adjust height as needed */
    overflow-y: auto;
    font-size: 2.8rem; /* Optional: adjust font size to fit more lines */
  }
</style>
```py
import networkx as nx
import matplotlib.pyplot as plt

# ----------------------------
# 1. Build a Larger, Layered Supply Chain Network
# ----------------------------
G = nx.DiGraph()

# Define layers
suppliers = [f"Supplier_{i}" for i in range(1, 6)]          # 5 suppliers
manufacturers = [f"Manufacturer_{i}" for i in range(1, 5)]   # 4 manufacturers
distributors = [f"Distributor_{i}" for i in range(1, 4)]     # 3 distributors
retailers = [f"Retailer_{i}" for i in range(1, 6)]          # 5 retailers

# Add all nodes
all_nodes = suppliers + manufacturers + distributors + retailers
G.add_nodes_from(all_nodes)

# Connect suppliers → manufacturers (each manufacturer uses 2–3 suppliers)
import random
random.seed(42)
for m in manufacturers:
    chosen_suppliers = random.sample(suppliers, k=random.randint(2, 3))
    for s in chosen_suppliers:
        G.add_edge(s, m)

# Connect manufacturers → distributors (each distributor gets from 2–3 manufacturers)
for d in distributors:
    chosen_manufacturers = random.sample(manufacturers, k=random.randint(2, 3))
    for m in chosen_manufacturers:
        G.add_edge(m, d)

# Connect distributors → retailers (each retailer served by 1–2 distributors)
for r in retailers:
    chosen_distributors = random.sample(distributors, k=random.randint(1, 2))
    for d in chosen_distributors:
        G.add_edge(d, r)

print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ----------------------------
# 2. Compute Centrality Metrics
# ----------------------------
pagerank = nx.pagerank(G, alpha=0.85)
betweenness = nx.betweenness_centrality(G.to_undirected())

# Combine scores to find most critical node
combined_scores = {node: pagerank[node] + betweenness[node] for node in G.nodes()}
most_critical = max(combined_scores, key=combined_scores.get)

print(f"\nMost critical node: {most_critical}")
print(f"  PageRank: {pagerank[most_critical]:.4f}")
print(f"  Betweenness: {betweenness[most_critical]:.4f}")

# ----------------------------
# 3. Simulate Disruption
# ----------------------------
G_disrupted = G.copy()
G_disrupted.remove_node(most_critical)

# Count customers who lost ALL their suppliers (in-degree = 0) after disruption
orphaned_after = [
    n for n in G_disrupted.nodes()
    if G_disrupted.in_degree(n) == 0 and n != most_critical
]

print(f"\nAfter removing {most_critical}:")
print(f"  Orphaned nodes (no incoming supply): {len(orphaned_after)}")
if orphaned_after:
    print("  Affected nodes:", ", ".join(orphaned_after))

# ----------------------------
# 4. Visualization
# ----------------------------
# Assign positions by layer for clarity
# Assign positions by layer for clarity
pos = {}
layer_spacing = 3

# Each entry: (list_of_nodes, y_level)
layers = [
    (suppliers, 0),
    (manufacturers, 1),
    (distributors, 2),
    (retailers, 3)
]

for node_list, y_level in layers:
    n = len(node_list)
    # Center the group horizontally
    x_positions = [i * layer_spacing for i in range(n)]
    x_center = (n - 1) * layer_spacing / 2
    for i, node in enumerate(node_list):
        pos[node] = (x_positions[i] - x_center, y_level)
        
# --- Plot Original vs Disrupted ---
plt.figure(figsize=(16, 8))

# Original Network
plt.subplot(1, 2, 1)
node_colors_orig = ['red' if node == most_critical else 'lightblue' for node in G.nodes()]
nx.draw(
    G, pos,
    node_color=node_colors_orig,
    node_size=900,
    with_labels=True,
    font_size=8,
    font_weight='bold',
    arrows=True,
    arrowsize=12,
    edge_color='gray',
    width=1.0
)
plt.title(f"Original Supply Chain\n(Red = Most Critical: {most_critical})")

# Disrupted Network
plt.subplot(1, 2, 2)
# Reuse same positions (skip missing node)
pos_disrupted = {n: pos[n] for n in G_disrupted.nodes()}

node_colors_disrupted = []
for node in G_disrupted.nodes():
    if node in orphaned_after:
        node_colors_disrupted.append('orange')      # lost all supply
    elif G_disrupted.out_degree(node) == 0 and G_disrupted.in_degree(node) > 0:
        node_colors_disrupted.append('lightgreen')  # end customer (retailer)
    else:
        node_colors_disrupted.append('lightblue')

nx.draw(
    G_disrupted, pos_disrupted,
    node_color=node_colors_disrupted,
    node_size=900,
    with_labels=True,
    font_size=8,
    font_weight='bold',
    arrows=True,
    arrowsize=12,
    edge_color='gray',
    width=1.0
)

# Add legend manually
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', label='Removed Node'),
    Patch(facecolor='orange', label='Orphaned (No Input)'),
    Patch(facecolor='lightgreen', label='End Customer'),
    Patch(facecolor='lightblue', label='Operational')
]
plt.legend(handles=legend_elements, loc='upper right')

plt.title(f"After Removing {most_critical}\n(Orange = Orphaned Nodes)")
plt.tight_layout()
plt.show()
```

