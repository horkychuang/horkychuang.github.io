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

![bg right fit](I:\My Drive\保留區\1142\1142 SNA\SNA1142\Class31\images\02201.jpg)

-----

## K-Core

- The **k-core** of a graph is the **maximal subgraph** in which **every node has degree ≥ k** (within the subgraph).
- It is obtained by **recursively removing nodes with degree < k**, until no such nodes remain.
- The **core number** of a node = the **highest k** for which the node belongs to the k-core.

-----

## Social media analysis

`Misinformation Superspreaders` via k-core 通過 k-core 識別虛假資訊超級傳播者

- **Network**: Users as nodes; retweets, shares, or mentions as edges.
- **k-core use**: High k-core nodes (e.g., users in the 10-core or higher) are deeply embedded in dense interaction clusters.
- **Insight**: These users are not just highly connected—they’re in **densely interconnected communities**, making them efficient **superspreaders** of misinformation. Removing them disrupts misinformation cascades more effectively than targeting high-degree nodes alone.

---

###### Why k-core works so well for finding misinformation superspreaders

In misinformation-spreading networks (retweet, quote, or reply networks of false/misleading posts), the structure is extremely skewed 在虛假資訊傳播網路（如轉發、引用或回復虛假/誤導性帖子所構成的網路）中，結構極度偏斜：

- A tiny number of accounts are responsible for the vast majority of diffusion. 極少數帳號製造了絕大多數資訊擴散
- These “superspreaders” sit in the densely connected core of the network. 這些“超級傳播者”位於網路中連接最密集的核心區域
- Peripheral users retweet once or twice and disappear. 週邊使用者往往只轉發一兩次便不再出現

---

The **k-core decomposition** perfectly captures this:

- The k-core is the maximal subgraph where every node has degree ≥ k. k-core 是指一個極大子圖，其中每個節點的度數（連接數）均不小於 k
- The **main core** (highest k such that the k-core is non-empty) contains almost exclusively the superspreaders. **主核**（即 k 值最大且 k-core 非空的核心）幾乎完全由超級傳播者構成；
- In practice, on Twitter/Facebook misinformation datasets, the main k-core typically has only 5–200 accounts but accounts for 60–95 % of all retweet volume. 在實際應用中，針對 Twitter 或  Facebook 的虛假資訊資料集，主 k-core 通常僅包含 5 至 200 個帳號，卻貢獻了 60% 至 95% 的全部轉發量。

---

## Finance

`Credit Risk in Guarantee Networks` (k-Core & Clustering) 擔保網路中的信用風險：基於 k-Core 與聚類分析

- **Network**: Firms as nodes; guarantee relationships (e.g., Firm A guarantees Firm B’s loan) as directed or undirected edges.
- **k-core use**: Firms in high k-cores are **mutually entangled** in dense guarantee clusters.
- **Insight**: A default by one firm in a high k-core can trigger **systemic risk** through cascading failures. Regulators use k-core to identify systemically risky clusters beyond simple debt size.

在中小企業融資中，**互保、聯保**（即企業之間互相為對方貸款提供擔保）是一種常見增信方式。In SME financing, **mutual guarantees** and **joint guarantees**—where firms cross-guarantee each other’s loans—are common credit enhancement mechanisms.然而，這種結構在經濟下行時可能演變為**風險傳染的加速器**——一家企業違約，可能通過擔保鏈條引發連鎖代償，最終導致整個擔保圈“集體爆雷”。

---

​    During economic downturns, this structure can become an **accelerator of risk contagion**: the default of a single firm may trigger a chain of guarantee obligations through the guarantee links, potentially causing an entire guarantee cluster to collapse simultaneously (“collective default”). 為識別此類高風險結構，可將擔保關係建模為一個**擔保網路**（Guarantee Network），並借助圖論工具進行系統性風險分析。

---

To identify such high-risk structures, guarantee relationships can be modeled as a **guarantee network**, and graph-theoretic tools can be leveraged for systemic risk analysis

- Build a guarantee graph, nodes = firms; a directed edge `A → B` indicates firm A guarantees B’s loan (or undirected edges for mutual guarantees). 節點 = 企業, 有向邊 `A `→`B` = 企業 A 為 B 的貸款提供擔保（或無向邊表示互保）
- Use **k-core decomposition** to find densely inter-guaranteed groups, i.e., tightly knit “guarantee cliques” or “guarantee chains.” Higher k-values indicate denser internal connectivity and greater potential for risk contagion. 找出**高度互聯、密集互保的子群**（即“擔保圈”或“擔保鏈”）, k 值越高，子群內部連接越緊密，風險傳染潛力越強

---

- Simulate **default cascades** within high-risk cores, 假設核心企業違約，觀察風險如何在高 k-Core 子圖中擴散, 評估局部風險是否會升級為系統性危機 Simulate **default cascades** within high-risk cores: assume a core firm defaults and observe how risk propagates through the high k-core subgraph to assess whether localized distress could escalate into a systemic crisis.
- Identify **systemically important guarantors**—firms that, even if financially sound, pose outsized systemic risk due to their **broad connectivity or central position** in the network. These can be detected using metrics such as **degree centrality, betweenness centrality, or k-core level**.  **識別系統重要性擔保人**（Systemically Important Guarantors）, 即使自身財務穩健，但因其**連接廣泛或位於核心位置**，一旦出險將引發大面積代償, 可通過**度中心性、介數中心性或** **k-Core** **層級**識別 

---

在擔保網路中：

- **Low k-core** (e.g., k=1): peripheral firms with limited risk impact. **低** **k-Core**（如 k=1）：邊緣企業，風險影響有限；
- **High k-core** (e.g., k=5): tightly coupled groups such as “guarantee triads” or closed-loop mutual guarantee clusters, where risks are highly interdependent. **高** **k-Core**（如 k=5）：形成“擔保鐵三角”或“閉環互保集團”，內部風險高度耦合；
- Default by any single firm within a high k-core can easily trigger a **domino effect**. 一旦高 k-Core 中任一企業違約，極易觸發**多米諾骨牌效應**。

---

## Marketing

### **Viral Product Launch or Influencer Campaign**

- **Network**: Customers or social media users as nodes; follows, likes, shares, or co-purchase links as edges.

- **k-core use**: Identify users in high k-cores (e.g., 8-core+) within a brand’s community graph.

- Insight: These users are part of tightly knit, highly interactive groups  (e.g., loyal fan communities, niche enthusiast circles).

  - **Marketing strategy**: Seed new products or campaigns to high k-core users—they’re more likely to generate **authentic, sustained word-of-mouth** within influential subcommunities than peripheral influencers with many but shallow connections.
  - **Why better than degree alone?** A high-degree user might have many followers but low engagement; a high k-core user is surrounded by active peers, amplifying message retention and spread.

----

- Builds a synthetic social network (using a **configuration model** with power-law degree distribution to mimic real social graphs).
- Computes the **k-core decomposition**.
- Compares diffusion from **top k-core nodes** vs. **top degree nodes** using a simple **Independent Cascade Model (ICM)**.
- Visualizes the network and plots diffusion results.

----

## Community Detection

1. Modularity Optimization:

- Modularity (Q) is a scalar metric that quantifies the quality of a partition. It's defined as:

$$
Q = \frac{1}{2m} \sum_{ij} \left( A_{ij} - \frac{k_i k_j}{2m} \right) \delta(c_i, c_j)
$$

Interpretation: Q compares the actual density of within-community edges to what would be expected in a random graph. Values range from -1 to 1; higher Q (e.g., >0.3) indicates strong community structure.

-----

###### 2. **Girvan–Newman Algorithm**

- **Hierarchical divisive** method. Repeatedly **remove edge with highest betweenness**.
- Communities emerge as graph breaks apart.

###### 3. **Louvain Method (Fast & Popular)**

- **Greedy optimization** of modularity. Very fast; works on large networks.
- Two phases repeated: Optimize modularity by moving nodes locally; build meta-graph where each community = node.

---

## **Social Media Analysis**

**Social Media: Misinformation or Polarization**

- **Use modularity** to confirm you’re not just splitting noise.
- Also compute:
  - **Conductance**: Measures how "leaky" a community is — lower = more isolated (stronger echo chamber)
  - **Homophily**: Do members share political/ideological signals?

###### These Metrics Reveal

| Metric                      | Interpretation in Polarization Context                       |
| --------------------------- | ------------------------------------------------------------ |
| **Conductance ≈ 0.1–0.2**   | Strong echo chamber — little exposure to opposing views      |
| **Conductance > 0.4**       | "Bridge" community — may include moderates or cross-ideological users |
| **Homogeneity > 90%**       | Detected community closely matches ideological ground truth  |
| **Network Homophily > 85%** | Confirms the network is highly polarized by design           |

---

## 

**Goal**: Understand information flow, polarization, and influence.

Key Applications:

- **Detecting echo chambers & filter bubbles**
  Communities often align with ideological, political, or interest-based groups. Identifying them helps assess misinformation spread or polarization (e.g., pro-vaccine vs. anti-vaccine clusters on Twitter).
- **Influencer identification within niches**
  Instead of global influencers, find **local leaders** in each community (e.g., a fitness micro-influencer in a health-conscious subnetwork).
- **Bot and coordinated inauthentic behavior detection**
  Artificially dense or anomalous communities may signal bot networks or troll farms.

*Example*: During elections, analysts use Louvain or Infomap to map communities sharing election-related content—revealing foreign interference or organic grassroots movements.

---

### **Detecting Polarized Communities**

This script simulates a **polarized social network** (e.g., pro-vaccine vs. anti-vaccine) and detects communities. It highlights how structural separation enables misinformation to thrive in isolated clusters.

---

## **Marketing**

### These Metrics Mean for Marketers

| Metric                              | Why It Matters                                               |
| ----------------------------------- | ------------------------------------------------------------ |
| **Conversion Lift**                 | Quantifies ROI of personalization. >50% lift is common in well-segmented campaigns. |
| **Cohort Retention Rate**           | Shows long-term value. High retention = lower CAC, higher LTV. |
| **Engagement Similarity (Jaccard)** | Measures behavioral cohesion. High similarity → messages spread faster and feel more relevant. |

----

**Goal**: Personalize outreach, optimize targeting, and foster organic advocacy.

#### Key Applications:

- **Segmenting audiences by behavioral cohesion**
  Communities often reflect shared lifestyles, values, or consumption patterns—even without demographic data. This enables **behavioral segmentation** beyond traditional RFM (Recency, Frequency, Monetary).
- **Seeding viral campaigns effectively**
  Launch a product within a tightly knit community (e.g., eco-conscious parents) rather than broadcasting broadly. High intra-community trust boosts conversion.
- **Identifying brand advocates & detractors**
  Detect communities where your brand is frequently discussed positively or negatively. Engage advocates; address detractors proactively.
- **Co-creation & community-driven innovation**
  Invite core members of relevant communities to beta-test or co-design products (e.g., gaming communities for new game features).

*Example*: A cosmetics brand detects a community of users frequently sharing DIY skincare tips. They partner with central members to launch a user-generated “natural routine” campaign—resulting in 3× higher engagement than generic ads.



----

###### Community-Based Customer Segmentation

This script simulates an **engagement network** from a social campaign and shows how to:

- Detect communities, Profile each by behavior, Recommend tailored messaging.

----

### **Community-Based Loyalty Segmentation**

This script simulates a **customer co-engagement network** and shows how to use community detection to design targeted retention strategies.

---

## **Finance**

✅ **Conductance** (to measure isolation of guarantee rings)
✅ **Anomaly score** based on **local density + size + isolation**
✅ **Fixed and completed visualization**
✅ **Clear risk ranking**

| Metric                 | Interpretation                                               |
| ---------------------- | ------------------------------------------------------------ |
| **High local density** | Firms are over-guaranteeing each other → mutual exposure     |
| **Low conductance**    | The group is isolated → shock won’t be absorbed by the wider network |
| **Anomaly score**      | Quantifies how much a community deviates from normal SME behavior |

---

**Goal**: Manage systemic risk, detect fraud, and uncover hidden relationships.

#### Key Applications:

- **Systemic risk in interbank or guarantee networks**
  Communities may represent tightly coupled financial groups (e.g., firms cross-guaranteeing loans). A shock in one community can cascade internally before spilling over.
- **Fraud ring detection**
  In transaction or account networks, fraudsters often form dense, isolated subgraphs (e.g., money mules sharing addresses/phones). Community detection flags these anomalous clusters.
- **Client segmentation for wealth management**
  Cluster clients based on transaction patterns, referral networks, or investment behaviors—enabling personalized advisory services.
- **Supply chain & counterparty risk**
  In corporate ownership or supply networks, communities may reveal hidden dependencies (e.g., multiple firms relying on the same obscure supplier).

*Example*: A bank applies Leiden algorithm to its corporate loan guarantee network and discovers a hidden community of 12 SMEs all guaranteeing each other’s loans. One defaults → high contagion risk → bank increases collateral requirements preemptively.



### **Detecting Risky Guarantee Communities**

This script builds a **corporate guarantee network**, detects communities, and flags high-risk clusters based on **density and size**—a common approach in systemic risk analysis.

---

| Metric                      | Why It Matters in Fraud                                      |
| --------------------------- | ------------------------------------------------------------ |
| **Local Density**           | Fraud rings simulate activity via internal transactions → abnormally high edge density |
| **Conductance (Isolation)** | Real businesses interact externally; fraud rings are closed loops → low conductance |
| **Anomaly Score**           | Combines multiple signals into a single risk metric — mimics real ML-based fraud systems |

### **Detecting Fraud Rings via Community Detection**

This script simulates a **simplified transaction network** with one synthetic fraud ring and many legitimate merchants. It uses community detection to flag high-risk clusters based on **anomalous density and isolation**. 

**Anomaly score per community** based on:

 - Density, Isolation (conductance), Size deviation from typical merchant clusters

---