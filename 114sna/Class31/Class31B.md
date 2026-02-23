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

**Class 031B Network Segment** 

**國企 Wen-Bin Chuang**
**2026-02-14**

![bg right fit](images\02201.jpg)

-----

## K-Core
<style scoped>
section {
    font-size: 25px;
}
</style>
- The **k-core** of a graph is the **maximal subgraph** in which **every node has degree ≥ k** (within the subgraph).圖的 k-核（k-core）是指圖中滿足以下條件的最大子圖：該子圖中的每個節點的度數均 ≥ k（度數僅計算子圖內部的連接）。
- It is obtained by **recursively removing nodes with degree < k**, until no such nodes remain.•	它可以通過遞迴地移除度數 < k 的節點來獲得，直到不再存在此類節點為止。
- The **core number** of a node = the **highest k** for which the node belongs to the k-core.•	一個節點的核數（core number）等於該節點所屬的 k-核中最大的 k 值。

---

## Social media analysis
<style scoped>
section {
    font-size: 25px;
}
</style>
#### Misinformation Superspreaders via k-core 通過 k-core 識別虛假資訊超級傳播者

In misinformation-spreading networks (retweet, quote, or reply networks of false/misleading posts), the structure is extremely skewed 在虛假資訊傳播網路（如轉發、引用或回復虛假/誤導性帖子所構成的網路）中，結構極度偏斜：

- A tiny number of accounts are responsible for the vast majority of diffusion. 極少數帳號製造了絕大多數資訊擴散
- These “superspreaders” sit in the densely connected core of the network. 這些“超級傳播者”位於網路中連接最密集的核心區域
- Peripheral users retweet once or twice and disappear. 週邊使用者往往只轉發一兩次便不再出現

---

The **k-core decomposition** perfectly captures this
<style scoped>
section {
    font-size: 25px;
}
</style>
- The **main core** (highest k such that the k-core is non-empty) contains almost exclusively the superspreaders. **主核**（即 k 值最大且 k-core 非空的核心）幾乎完全由超級傳播者構成；
- In practice, on Twitter/Facebook misinformation datasets, the main k-core typically has only 5–200 accounts but accounts for 60–95 % of all retweet volume. 在實際應用中，針對 Twitter 或  Facebook 的虛假資訊資料集，主 k-core 通常僅包含 5 至 200 個帳號，卻貢獻了 60% 至 95% 的全部轉發量。

---

## Finance

###### Credit Risk in Guarantee Networks (k-Core & Clustering) 擔保網路中的信用風險
<style scoped>
section {
    font-size: 25px;
}
</style>
In SME financing, **mutual guarantees** and **joint guarantees**—where firms cross-guarantee each other’s loans—are common credit enhancement mechanisms. 在中小企業融資中，**互保、聯保**（即企業之間互相為對方貸款提供擔保）是一種常見增信方式。



​    During economic downturns, this structure can become an **accelerator of risk contagion**: the default of a single firm may trigger a chain of guarantee obligations through the guarantee links, potentially causing an entire guarantee cluster to collapse simultaneously (“collective default”). 然而，這種結構在經濟下行時可能演變為**風險傳染的加速器**——一家企業違約，可能通過擔保鏈條引發連鎖代償，最終導致整個擔保圈“集體爆雷”。



  To identify such high-risk structures, guarantee relationships can be modeled as a **guarantee network**, and graph-theoretic tools can be leveraged for systemic risk analysis 為識別此類高風險結構，可將擔保關係建模為一個**擔保網路**（Guarantee Network），並借助圖論工具進行系統性風險分析。

---
<style scoped>
section {
    font-size: 25px;
}
</style>
Use **k-core decomposition** to find densely inter-guaranteed groups, i.e., tightly knit “guarantee cliques” or “guarantee chains.” Higher k-values indicate denser internal connectivity and greater potential for risk contagion. 找出**高度互聯、密集互保的子群**（即“擔保圈”或“擔保鏈”）, k 值越高，子群內部連接越緊密，風險傳染潛力越強



Identify **systemically important guarantors**—firms that, even if financially sound, pose outsized systemic risk due to their **broad connectivity or central position** in the network. These can be detected using metrics such as **degree centrality, betweenness centrality, or k-core level**.  **識別系統重要性擔保人**（Systemically Important Guarantors）, 即使自身財務穩健，但因其**連接廣泛或位於核心位置**，一旦出險將引發大面積代償, 可通過**度中心性、介數中心性或** **k-Core** **層級**識別 

---

在擔保網路中：
<style scoped>
section {
    font-size: 25px;
}
</style>
- **Low k-core** (e.g., k=1): peripheral firms with limited risk impact. **低** **k-Core**（如 k=1）：邊緣企業，風險影響有限；
- **High k-core** (e.g., k=5): tightly coupled groups such as “guarantee triads” or closed-loop mutual guarantee clusters, where risks are highly interdependent. **高** **k-Core**（如 k=5）：形成“擔保鐵三角”或“閉環互保集團”，內部風險高度耦合；
- Default by any single firm within a high k-core can easily trigger a **domino effect**. 一旦高 k-Core 中任一企業違約，極易觸發**多米諾骨牌效應**。

---

## Marketing
<style scoped>
section {
    font-size: 25px;
}
</style>
#### **Viral Product Launch or Influencer Campaign** 病毒式產品發佈或網紅行銷活動

- **k-core use**: Identify users in high k-cores (e.g., 8-core+) within a brand’s community graph.在品牌社群關係圖中識別處於高 k-core（例如 8-core 及以上）的用戶

- Insight: These users are part of tightly knit, highly interactive groups  (e.g., loyal fan communities, niche enthusiast circles). 這些用戶身處緊密聯結、高度互動的群體之中（例如忠實粉絲社群、小眾愛好者圈子）

  - **Marketing strategy**: Seed new products or campaigns to high k-core users—they’re more likely to generate **authentic, sustained word-of-mouth** within influential subcommunities than peripheral influencers with many but shallow connections. 將新產品或行銷活動優先投放給高 k-core 用戶——相較于那些擁有大量但淺層連接的邊緣網紅，他們更有可能在具有影響力的子社群中激發真實且持續的口碑傳播
  - **Why better than degree alone?** A high-degree user might have many followers but low engagement; a high k-core user is surrounded by active peers, amplifying message retention and spread. 高度數用戶可能擁有眾多關注者，但互動性低；而高 k-core 用戶周圍環繞著活躍的同儕，能顯著增強資訊的記憶度與擴散效果。

---

## Community Detection 社區發現
<style scoped>
section {
    font-size: 25px;
}
</style>
1. Modularity Optimization 模組度優化: Modularity (Q) is a scalar metric that quantifies the quality of a partition.模組度（Q）是一種標量指標，用於衡量網路劃分的品質。模組度越高，表示社區內部連接越緊密，社區之間連接越稀疏

2. Girvan–Newman Algorithm: Hierarchical divisive method. Repeatedly **remove edge with highest betweenness**. 一種層次化的分裂式方法。通過反復移除邊介數（betweenness）最高的邊，逐步將網路分解為多個社區。

3. **Louvain Method (Fast & Popular)**: Greedy optimization of modularity. Very fast; works on large networks. 採用貪心策略優化模組度，計算效率高，適用於大規模網路，在實際應用中廣受歡迎。



---

## Social Media Analysis
<style scoped>
section {
    font-size: 23px;
}
</style>
- **Detecting echo chambers & filter bubbles** **識別資訊繭房與回音室效應**
  Communities often align with ideological, political, or interest-based groups. Identifying them helps assess misinformation spread or polarization (e.g., pro-vaccine vs. anti-vaccine clusters on Twitter). 社群往往與意識形態、政治立場或興趣群體高度一致。識別這些社群有助於評估虛假資訊的傳播路徑或社會極化程度（例如在 Twitter 上支持疫苗與反對疫苗的用戶集群）。

- **Influencer identification within niches** **在細分領域中識別關鍵影響者**

  Instead of global influencers, find **local leaders** in each community (e.g., a fitness micro-influencer in a health-conscious subnetwork). 與其依賴全域性網紅，不如在每個社群內部發掘**本地意見領袖**（例如在注重健康的子網路中，一位專注於健身的微型影響者）。


- **Bot and coordinated inauthentic behavior detection** **識別機器人帳號與協同性不實行為**
  Artificially dense or anomalous communities may signal bot networks or troll farms. 結構異常密集或不符合自然交互模式的社群，可能暗示存在機器人網路或水軍組織。

---

###### Conductance and Homogeneity
<style scoped>
section {
    font-size: 23px;
}
</style>
**Conductance** and **Homogeneity** are two widely used metrics to evaluate the quality of a detected community (or cluster) in a graph. They assess different aspects of community structure

**Conductance**割比

Measures how **well-separated** a community is from the rest of the graph. 衡量一個社區與圖中其餘部分的**分離程度**，僅依賴圖的**拓撲結構**（無需真實標籤）

- **Low conductance** (close to 0) → Community is **well-separated** (few external edges relative to its internal connectivity). **越低越好**（接近 0）→ 社區內部緊密，與外部連接稀疏（**高品質社區**）**Conductance ≈ 0.1–0.2** lower conductance --> Echo Chamber


- **High conductance** (close to 1) → Community is **poorly isolated** (many external connections). **越高越差**（接近 1）→ 社區與外部高度連接，邊界模糊

To evaluate a **full partition**, average conductance across all communities (or report min/max). 若評估整個劃分，可對所有社區的 conductance 取平均

---
<style scoped>
  pre {
    max-height: 400px; /* Adjust height as needed */
    overflow-y: auto;
    font-size: 2.8rem; /* Optional: adjust font size to fit more lines */
  }
</style>
```py
# Compute Conductance for each detected community
def conductance(G, community):
    """Conductance = cut_edges / vol(community)
       Lower = more isolated (stronger community)"""
    community_set = set(community)
    cut_edges = 0
    vol = 0
    for node in community:
        for neighbor in G.neighbors(node):
            vol += 1
            if neighbor not in community_set:
                cut_edges += 1
    return cut_edges / vol if vol > 0 else 1.0

for i, comm in enumerate(communities):
    cond = conductance(G, comm)   # G is the original
```



---

**Homogeneity**同質性 
<style scoped>
section {
    font-size: 25px;
}
</style>
Measures whether a community contains nodes from **only one ground-truth class** 衡量一個社區是否只包含**同一類真實標籤**的節點。**需要真實標籤**。

- A clustering is **homogeneous** if all nodes in a cluster belong to the **same class** 如果每個檢測出的社區中，所有節點都來自**同一個真實類別**，則同質性為 1。

- **Homogeneity = 1**: Perfect (each community has nodes from exactly one true class). **越高越好**（1 = 完美同質）。
- **Homogeneity = 0**: Worst case (community mixes many classes). 0 = 社區混合了多個真實類別。

---
<style scoped>
  pre {
    max-height: 400px; /* Adjust height as needed */
    overflow-y: auto;
    font-size: 2.8rem; /* Optional: adjust font size to fit more lines */
  }
</style>
```py
Compute Homophily (based on ground-truth labels)
def homophily(G, label_dict):
    """Fraction of edges where both endpoints share the same label."""
    same_label_edges = 0
    total_edges = G.number_of_edges()
    for u, v in G.edges():
        if label_dict[u] == label_dict[v]:
            same_label_edges += 1
    return same_label_edges / total_edges if total_edges > 0 else 1.0


for i, comm in enumerate(communities):    
    # Homogeneity: % of nodes in this community sharing the same true label
    labels_in_comm = [true_label[node] for node in comm]
    most_common_label = max(set(labels_in_comm), key=labels_in_comm.count)
    homogeneity = labels_in_comm.count(most_common_label) / len(labels_in_comm)
    
# Overall network homophily
overall_homophily = homophily(G, true_label)
```

---

#### Metrics Reveal
<style scoped>
section {
    font-size: 25px;
}
</style>
| Metric                      | Interpretation in Polarization Context                       |
| --------------------------- | ------------------------------------------------------------ |
| **Conductance ≈ 0.1–0.2**   | Strong echo chamber — little exposure to opposing views      |
| **Conductance > 0.4**       | "Bridge" community — may include moderates or cross-ideological users |
| **Homogeneity > 90%**       | Detected community closely matches ideological ground truth  |
| **Network Homophily > 85%** | Confirms the network is highly polarized by design           |



---

`Low conductance + higher degree`
<style scoped>
section {
    font-size: 25px;
}
</style>
- **回音室**（Echo Chamber）：Users are exposed only to similar viewpoints and reject external information 使用者只接觸相似觀點，排斥外部資訊 → leading to polarization and the spread of rumors → 極化、謠言擴散。
  - lower conductance --> echo chamber --> this community is highly insulated
- **Hub**（樞紐用戶）：Highly connected individuals serve as critical nodes in information dissemination高連接度個體是資訊傳播的關鍵節點。
  - higher degree --> hub--> most well-connected user

---
<style scoped>
section {
    font-size: 25px;
}
</style>
By identifying a community that is both **highly insulated** (low conductance) and **dominated by skeptics** (>80%), and then locating the **most well-connected user** within it, we can pinpoint a key individual who may be driving the spread of misinformation. 通過識別一個**高度封閉**（低conductance）且**以懷疑論者為主**（>80%）的社區，並找出其中**連接最廣的用戶**，來定位可能推動虛假資訊傳播的關鍵人物。

- **節點** **ID ≥ 100**：Represents "skeptics"—users who distrust mainstream information and tend to share alternative or contrarian viewpoints.代表“懷疑論者”（skeptics），例如不相信主流資訊、傾向於傳播另類觀點的用戶。

- **節點** **ID < 100**：Represents "regular" or "non-skeptic" users 代表“普通用戶”或“非懷疑論者”。

---

###### Finding the central user (Hub) within this community在該社區中尋找中心用戶（Hub）
<style scoped>
section {
    font-size: 25px;
}
</style>
- Extract the **induced subgraph** of the community. 提取該社區的**誘導子圖**（induced subgraph）。
- Compute the **degree** of each node in this subgraph—i.e., how many connections each user has within the community. 計算子圖中每個節點的**度**（degree）—— 即該用戶在社區內部有多少連接。
- Select the node with the **highest degree** as the **potential misinformation hub**, because highly connected users wield greater influence inside the community. 選擇**度最大**的節點作為 潛在虛假資訊樞紐（**hub**）


If they disseminate false information, it spreads rapidly and widely.若其傳播錯誤資訊，擴散效率高。

If a **closed community** (low conductance) contains a **high-influence ** (hub), that user is likely to act as an **amplifier of false or misleading information**. 若一個**封閉社區**（低 conductance）中存在一個**高影響力**（hub），那麼該用戶很可能成為**虛假或誤導性資訊的放大器**。

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
import random

# 1. Build a polarized network: two echo chambers + a few bridge users
G = nx.Graph()
G.add_nodes_from(range(200))  # 0–99: Pro-Vax, 100–199: Skeptical

random.seed(42)
# Dense internal connections
for i in range(100):
    for j in range(i+1, 100):
        if random.random() < 0.08:
            G.add_edge(i, j)

for i in range(100, 200):
    for j in range(i+1, 200):
        if random.random() < 0.07:
            G.add_edge(i, j)

# Add bridge edges
for _ in range(8):
    a = random.randint(0, 99)
    b = random.randint(100, 199)
    G.add_edge(a, b)

print(f"Network built: {G.number_of_nodes()} users, {G.number_of_edges()} interactions")

# Ground-truth labels (for homophily)
true_label = {node: 0 if node < 100 else 1 for node in G.nodes()}  # 0=Pro, 1=Skeptical

# 2. Detect communities
communities = nx.community.louvain_communities(G, seed=42)
modularity = nx.community.modularity(G, communities)
print(f"\nLouvain found {len(communities)} communities.")
print(f"Modularity: {modularity:.3f}")


def conductance(G, community):
    """Fraction of edges leaving the community vs. total edges incident to it."""
    comm_set = set(community)
    cut_edges = 0
    total_edges = 0
    for node in community:
        for neighbor in G.neighbors(node):
            total_edges += 1
            if neighbor not in comm_set:
                cut_edges += 1
    return cut_edges / total_edges if total_edges > 0 else 1.0

def homophily(G, label_dict):
    """Fraction of edges connecting nodes with the same label."""
    same = 0
    total = G.number_of_edges()
    for u, v in G.edges():
        if label_dict[u] == label_dict[v]:
            same += 1
    return same / total if total > 0 else 1.0

# 3. Analyze communities with conductance and homogeneity
print("\n Community Analysis (with Conductance & Homogeneity):")
for i, comm in enumerate(communities):
    pro = sum(1 for node in comm if node < 100)
    skept = len(comm) - pro
    size = len(comm)
    
    # Conductance
    cond = conductance(G, comm)
    
    # Homogeneity: % of dominant group in this community
    dominant_frac = max(pro, skept) / size
    
    print(f"  Community {i}: {pro} pro-vax, {skept} skeptical "
          f"| Conductance={cond:.3f} | Homogeneity={dominant_frac:.2%}")

# Network-wide homophily
net_homophily = homophily(G, true_label)
print(f"\n Network homophily (ideological alignment): {net_homophily:.2%}")

# 4. Visualize
partition = {}
for i, comm in enumerate(communities):
    for node in comm:
        partition[node] = i

colors = [partition[node] for node in G.nodes()]
pos = nx.spring_layout(G, seed=42, k=0.4)

plt.figure(figsize=(10, 8))
nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.tab10, node_size=30)
nx.draw_networkx_edges(G, pos, alpha=0.3)
plt.title("Detected Communities in a Polarized Vaccine Debate Network", fontsize=13)
plt.axis("off")
plt.tight_layout()
plt.show()

# 5. Find misinformation hub in skeptical-dominant community
skeptical_community = None
for comm in communities:
    skept_count = sum(1 for node in comm if node >= 100)
    if skept_count > len(comm) * 0.8:
        skeptical_community = comm
        break

if skeptical_community:
    sub = G.subgraph(skeptical_community)
    degrees = dict(sub.degree())
    hub = max(degrees, key=degrees.get)
    hub_cond = conductance(G, skeptical_community)
    print(f"\n  Potential misinformation hub: User {hub}")
    print(f"    → In skeptical-dominant community (size={len(skeptical_community)})")
    print(f"    → Degree: {degrees[hub]}, Community conductance: {hub_cond:.3f}")
    print("    → Low conductance suggests this cluster is an insulated echo chamber.")
```



---

## Finance
<style scoped>
section {
    font-size: 25px;
}
</style>
- **Systemic risk in interbank or guarantee networks** **銀行間或擔保網路中的系統性風險**
  Communities may represent tightly coupled financial groups (e.g., firms cross-guaranteeing loans). A shock in one community can cascade internally before spilling over. 社群可能代表高度關聯的金融群體（例如相互提供貸款擔保的企業）。一旦某一社群內部發生衝擊，風險會先在內部迅速傳導，隨後才向外溢出。

- **Fraud ring detection** **欺詐夥識別**
  In transaction or account networks, fraudsters often form dense, isolated subgraphs (e.g., money mules sharing addresses/phones). Community detection flags these anomalous clusters. 在交易或帳戶網路中，欺詐者通常會形成密集且孤立的子圖（例如多個“錢騾”共用地址或電話號碼）。通過社群發現演算法可有效識別這些異常聚集。

---
<style scoped>
section {
    font-size: 25px;
}
</style>  
- **Client segmentation for wealth management** **財富管理中的客戶細分**
  Cluster clients based on transaction patterns, referral networks, or investment behaviors—enabling personalized advisory services. 基於客戶的交易模式、轉介關係或投資行為進行聚類，從而提供個性化理財顧問服務。

- **Supply chain & counterparty risk** **供應鏈與交易對手風險**
  In corporate ownership or supply networks, communities may reveal hidden dependencies (e.g., multiple firms relying on the same obscure supplier). 在企業股權或供應鏈網路中，社群結構可能揭示隱藏的依賴關係（例如多家公司共同依賴某一家冷門供應商）。

---

##### **Detecting Fraud Rings via Community Detection**通過社區發現檢測欺詐團夥
<style scoped>
section {
    font-size: 25px;
}
</style>
This script simulates a **simplified transaction network** with one synthetic fraud ring and many legitimate merchants. It uses community detection to flag high-risk clusters based on **anomalous density and isolation**. 一個**簡化的交易網路**，其中包含一個合成的欺詐團夥和眾多合法商戶。它利用社區發現技術，根據**異常的密集度與隔離性**來標記高風險集群。

---
<style scoped>
section {
    font-size: 25px;
}
</style>
**Anomaly score per community** based on: **每個社區的異常評分**基於以下指標

 - Density, Isolation (conductance), Size deviation from typical merchant clusters 密集度、隔離度（導通率）、以及社區規模與典型商戶集群的偏差
 - In the context of transaction network analysis, **density**, **conductance (as a measure of isolation)**, and **community size** are three **key structural features** used to **detect potential fraud rings** 在交易網路分析的背景下，**密集度**、**導通率**（作為隔離性的度量）和**社區規模**是用於**檢測潛在欺詐團夥**的三個**關鍵結構性特徵**

----
<style scoped>
section {
    font-size: 25px;
}
</style>
- Fraudsters often need to **create the illusion of legitimate business activity**. 欺詐者通常需要**製造合法商業活動的假像**。 For example:

  - In **merchant fraud**, fake stores might “sell” to each other using stolen cards to inflate sales volume and qualify for higher transaction limits or payouts. 在**商戶欺詐**中，虛假店鋪可能使用盜取的信用卡相互“銷售”，以虛增銷售額，從而獲得更高的交易額度或結算款項。
  - In **money laundering**, funds are cycled rapidly among colluding accounts to obscure their origin (“layering”). 在**洗錢**活動中，資金會在共謀帳戶之間快速迴圈，以掩蓋其來源（即“分層”操作）

----
<style scoped>
section {
    font-size: 25px;
}
</style>
#### **Network effect 網路效應:**
  This generates **many transactions within the ring**, resulting in a **high edge density**—far above what you’d see in a typical group of independent legitimate merchants (who transact more sparsely and with external partners).   這會在團夥內部產生**大量交易**，導致**極高的邊密集度**——遠高於典型的獨立合法商戶群體（後者交易更為稀疏，且多與外部夥伴進行）。In graph terms: the subgraph induced by the fraud ring approaches a clique (fully connected). 從圖論角度看，欺詐團夥所誘導的子圖接近於一個**團**（完全連接圖）。

  - **Density → "Too tightly connected?"**. Fraud rings create **artificial transaction loops** among themselves → **abnormally high density**   密集度 **→** “連接是否過於緊密？ 欺詐團夥在內部製造人為的交易迴圈** → 導致**異常高的密集度**。

---

Exposure increases risk**暴露會增加風險**:
<style scoped>
section {
    font-size: 25px;
}
</style>
- Every external transaction is a potential **audit trail**. Legitimate counterparties might report suspicious behavior. o  每一筆外部交易都可能成為一條**審計線索**。合法的交易對手方可能會舉報可疑行為。
- Linking too openly to the broader network makes the ring **easier to detect via connectivity patterns**. 若與更廣泛的網路連接過於明顯，該團夥會因**連接模式**而**更容易被發現**。

So fraud rings **minimize “leakage”**—they transact mostly among themselves, with only a few carefully placed external links (e.g., to cash out or blend in). 因此，欺詐團夥會儘量減少洩露——主要在內部進行交易，僅保留少數精心佈置的外部連結（例如用於套現或偽裝融入）。

---

#### **Network effect **網路效應**:**
<style scoped>
section {
    font-size: 25px;
}
</style>
This leads to **low conductance** (or high modularity): the fraction of edges leaving the group is small relative to internal edges.這導致了**低導通率**（或高模組度）：即離開該群體的邊占總邊數的比例很小。 The ring appears as an **isolated “island”** in the network.  該團夥在網路中表現為一個**孤立的島嶼”**。This isolation is a key red flag—legitimate businesses usually have diverse, external-facing transaction partners.這種隔離性是一個關鍵警示信號——合法企業通常擁有多樣化且面向外部的交易夥伴。

- **Conductance → "Too isolated from the rest of the network?"**.導通率 → “與網路其餘部分是否過於隔離？”  Fraction of a community’s edges that point *outside* the group. 指一個社區中指向*外部*的邊所占的比例。 Low conductance = few external links = high isolation. 低導通率 = 外部連結少 = 高度隔離。A community with **low conductance** (e.g., < 0.3) is behaving like a closed clique—typical of collusion  一個**導通率很低**（例如 < 0.3）的社區，其行為如同一個封閉的團，這通常是共謀的典型特徵

---

Fraud requires **trust and secrecy**: **欺詐活動依賴信任與保密**
<style scoped>
section {
    font-size: 25px;
}
</style>
- Larger groups increase the chance of **leaks, mistakes, or betrayal**. Coordination (e.g., timing fake transactions, sharing stolen card data) is simpler with fewer participants. 群體規模越大，發生**洩密、失誤或背叛**的可能性就越高。較少的參與者使得協調工作（例如安排虛假交易的時間、共用被盜信用卡資料）更為簡單。
- Small rings are **more agile** and less likely to attract statistical scrutiny (e.g., a 10-merchant ring looks less suspicious than a 100-merchant anomaly).   小型團夥**更加靈活**，也更不容易引起統計審查（例如，一個由10家商戶組成的團夥，比一個由100家商戶組成的異常集群看起來要可疑得多）。

---

#### **Network effect**網路效應
<style scoped>
section {
    font-size: 25px;
}
</style>
Fraud communities are often **compact** (e.g., 5–20 nodes), unlike legitimate business clusters that may organically grow larger through supply chains or customer networks.  欺詐社區通常是**緊湊的**（例如，5至20個節點），而不像合法的商業集群那樣，可能通過供應鏈或客戶網路自然增長到更大規模。Very small communities (size = 1 or 2) are usually ignored in detection, so fraudsters often pick the “sweet spot”: big enough to generate activity, small enough to stay covert. 在檢測中，規模極小的社區（如1或2個節點）通常會被忽略，因此欺詐者往往會選取一個“最佳規模”：大到足以產生足夠活動，又小到足以保持隱蔽。

- **Size → "Unusually small and tight?"**. Number of nodes in the community. Communities **significantly smaller than average** (especially if dense and isolated) raise suspicion. **規模 → “是否異常地小而緊密？”** 指社區中的節點數量。**顯著小於平均水準**的社區（尤其是當其同時具備高密集度和高隔離性時）會引起懷疑。

---

## Marketing
<style scoped>
section {
    font-size: 25px;
}
</style>
#### **Segmenting audiences by behavioral cohesion** **基於行為凝聚力進行細分**
  Communities often reflect shared lifestyles, values, or consumption patterns—even without demographic data. This enables **behavioral segmentation** beyond traditional RFM (Recency, Frequency, Monetary). 社群往往體現出共同的生活方式、價值觀或消費模式——即使缺乏人口統計資料，也能實現超越傳統RFM（最近購買時間、購買頻率、消費金額）模型的**行為細分**。

- **Seeding viral campaigns effectively** **高效啟動病毒式行銷活動**
  Launch a product within a tightly knit community (e.g., eco-conscious parents) rather than broadcasting broadly. High intra-community trust boosts conversion. 緊密聯結的社群內部（例如注重環保的群體）推出產品，而非廣泛撒網式宣傳。社群內部的高度信任可顯著提升轉化率。

---
<style scoped>
section {
    font-size: 25px;
}
</style>  
- **Identifying brand advocates & detractors** **識別品牌擁護者與批評者**
  Detect communities where your brand is frequently discussed positively or negatively. Engage advocates; address detractors proactively. 發現那些頻繁正面或負面討論你品牌的社群。主動與擁護者互動，同時積極應對批評者。


One core objective of marketing is to reach the users most likely to respond with the most relevant content.  行銷的核心目標之一是**用最相關的內容觸達最可能回應的用戶**。However, in practice, we often do not know each user's true interests (i.e., their *true persona*), or we cannot directly use such sensitive information due to privacy constraints.但現實中，我們通常**不知道每個用戶的真實興趣**（即 `true_persona`），或者不能直接使用這些敏感資訊（如隱私限制）。

A practical alternative is: first identify the group (or community) a user belongs to, then assign a "representative marketing persona" to that entire group, and finally deliver a single, unified yet customized message tailored to that persona. 一個實用的替代方案是: 先識別用戶所屬的群體（社區），再為整個群體賦予一個“代表性的行銷人設”，然後由這個行銷人設 發送某一種統一但定制化的消息。persona

---

####  Three Key Marketing Metrics 三項關鍵行銷指標 

###### Conversion Lift 轉化率提升
<style scoped>
section {
    font-size: 25px;
}
</style>
- Each user in a community has a **true persona** (`true_persona[node]`).對社區中的每個用戶都有其**真實人設**（`true_persona[node]`）。
  - **Personalized messaging**定制化消息: Based on the assigned marketing representative persona (e.g., “Urban Runners”) and the user’s true persona, we look up a relevance score from a predefined table. 根據分配的行銷代表人設（如“Urban Runners”）和用戶的真實人設，查表得到相關性得分，以此作為轉化概率進行隨機模擬。This score serves as the conversion probability in a stochastic simulation. 
  - **Generic messaging**是通用消息：A uniform baseline conversion rate of 30% is applied to all users統一使用 30% 的基準轉化率。
  - The **conversion lift (%)** is calculated as the percentage improvement of personalized messaging over generic messaging. 利用**定制化** **vs** **通用消息的轉化率差異**，並計算得出提升百分比（Lift %）：

---

##### Simulated User Retention模擬用戶留存 (Over 3 Periods) 
<style scoped>
section {
    font-size: 25px;
}
</style>
- We assume that the more relevant the message (i.e., the closer the match between the marketing persona and the user’s true persona), the slower the user churns. 假設消息越相關 （行銷代表人設和用戶的真實人設愈相近），用戶流失越慢。
- In each period `t` (over 3 periods total), we compute a retention probability. 在每個週期 `t`（共 3 期），計算留存概率. 
  - Base retention declines over time (0.9 → 0.8 → 0.7). 基礎留存隨時間下降（0.9 → 0.8 → 0.7）。If message relevance > 0.5, an additional boost is applied 如果消息相關性高（>0.5），則額外加分; if relevance ≤ 0.5, a penalty is applied. Retention never falls below 30%. ；低則扣分。最低不低於 30%。
  - The final metric recorded is the **cohort retention rate at period 3**. 最終記錄第 3 期的 群體留存率(cohort retention)。

---

##### Interaction Similarity Within Community 計算社區內用戶的互動相似性（Jaccard 係數）
<style scoped>
section {
    font-size: 26px;
}
</style>
- For all pairwise combinations of users within a community, compute the Jaccard similarity between their neighbor sets. 對社區內所有使用者兩兩組合，計算他們鄰居集合的 Jaccard 相似度。 
  - A higher average Jaccard similarity indicates a more tightly knit community with more homogeneous interaction patterns—i.e., higher community quality. (Note: this metric is independent of the marketing persona assignment.) 平均 Jaccard 值越高，說明社區內部結構越緊密、使用者互動模式越相似（即社區品質高）（這裡與行銷代表人設無關）
  - **Jaccard similarity** measures how similar members’ social adjacency structures are, reflecting the community’s internal cohesion or homogeneity. 通過 **Jaccard** 相似係數（Jaccard Similarity） 來衡量一個社區內部成員的社交鄰接結構有多相似，從而反映該社區的內在凝聚力或同質性。
    - Higher value → greater overlap in users’ “social circles” → tighter community structure. 值越高 → 用戶之間的“社交圈子”越重疊 → 社區越緊密。

---
<style scoped>
section {
    font-size: 25px;
}
</style>
Consider a community of 100 users, of whom 80 are truly “Zero-Waste Advocates.””假設有一個 100 人的社區：其中 80 人是真正的 “Zero-Waste Advocates”.  However, because the community size (100) falls within the range [95, 110], the `assign_persona` logic labels it as “Urban Runners.  但因為社區規模是 100（介於 95 和 110 之間），`assign_persona` 把它標為 “Urban Runners".；

The system then sends “runner-themed” ads to the entire group.於是系統向他們發送“跑者”主題的廣告；雖然消息對“跑者”很有效（0.70），但對“零廢棄宣導者”效果差（僅 0.20）；最終轉化率低，Lift 小Although this message is highly effective for actual runners (conversion = 0.70), it performs poorly for zero-waste advocates (conversion = 0.20). Consequently, overall conversion remains low and lift is minimal.

→ This reveals the **cost of misassigned personas** and underscores that high-quality community detection + accurate persona mapping = effective marketing. 這就暴露了**錯誤分配人設的代價**，也說明：**好的社區劃分** **+** **準確的人設映射** **=** **高效行銷**。

---

### Metrics Mean for Marketers
<style scoped>
section {
    font-size: 25px;
}
</style>
| Metric                              | Why It Matters                                               |
| ----------------------------------- | ------------------------------------------------------------ |
| **Conversion Lift**                 | Quantifies ROI of personalization. >50% lift is common in well-segmented campaigns. |
| **Cohort Retention Rate**           | Shows long-term value. High retention = lower CAC, higher LTV. |
| **Engagement Similarity (Jaccard)** | Measures behavioral cohesion. High similarity → messages spread faster and feel more relevant. |

---
<style scoped>
section {
    font-size: 25px;
}
</style>
Ultimately, the personalized conversion rate, generic conversion rate, and resulting lift are jointly determined by the **population composition** and the **conversion probabilities of each user type** under a given message. 定制化轉化率、通用轉化率以及最終的 lift（提升）本質上是由“人群結構”和“各類使用者的轉化概率”共同決定的

- Expected personalized conversion rate = ∑ (proportion of user type × conversion probability of that type under the assigned message). 期望轉化率 = $\sum$ (某類用戶占比 × 該類使用者對該消息的轉化概率)

- Expected generic conversion rate = constant baseline (e.g., 30%). 通用消息下的期望轉化率(假設為常數)

- Expected Lift (%) = [(Expected personalized conversion − Generic conversion) / Generic conversion] × 100%. 期望 Lift（轉化提升百分比）
  $$
  \text{Lift}=\frac{\text{Conv}_{tailored}-\text{Conv}_{generic}}{\text{Conv}_{generic}}
  $$







