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

**Class 031 Network Relation** 

**國企 Wen-Bin Chuang**
**2026-02-14**

![bg right fit](images\02201.jpg)

---

##  Network Relation I: Triadic Closure
<style scoped>
section {
    font-size: 23px;
}
</style>
Triadic closure refers to the tendency in social networks for two nodes with a common neighbor to form a direct connection, forming a triangle (triad). This is a key mechanism for network densification and clustering, often measured by `clustering coefficient` or `transitivity.` 三元閉包（Triadic closure）是指在社交網路中，兩個擁有共同鄰居的節點傾向于建立直接連接，從而形成一個三角形（即三元組）。這是網路密度增加和聚類形成的關鍵機制，通常通過 `聚類係數（clustering coefficient）`或`傳遞性（transitivity）`來衡量。


In social media, `high triadic closure` indicates strong community formation, like friend recommendations based on mutual followers.  在社交媒體中，`較高的三元閉包程度`表明社區結構緊密，例如基於共同關注者的好友推薦機制。

`High clustering` suggests dense communities (e.g., `echo chambers` on social media); `low values` indicate sparse connections ripe for growth via recommendations. `高聚類係數`意味著存在密集的社群（如社交媒體上的`回音室效應`）；`低聚類係數`則表明連接較為稀疏，存在通過推薦系統促進新連接增長的空間。



----
<style scoped>
section {
    font-size: 25px;
}
</style>
- Goals: identifies pairs of nodes (users) that are not directly connected but share one or more common neighbors (mutual friends). 識別那些尚未直接相連但擁有一個或多個共同鄰居（即共同好友）的節點對（用戶）。These pairs are the `open triads` or `open wedges` — the foundation of triadic closure theory. 這些節點對被稱為“開放三元組”（open triads）或“開放楔形結構”（open wedges），是三元閉包理論的基礎。
  - If two people have mutual friends, they are likely to become friends themselves. 如果兩個人有共同好友，他們很可能彼此成為好友。
  - The more mutual friends they share, the higher the likelihood of a new connection forming. 共同好友越多，形成新連接的可能性就越高。
  - Social media platforms use this exact logic for friend/follower recommendations (e.g., "People You May Know") 社交媒體平臺正是基於這一邏輯進行好友/關注者推薦（例如“你可能認識的人”)

It demonstrates triadic closure — the principle that if two people share a common friend, they are likely to become friends themselves (forming a "triad" or triangle). 這體現了三元閉包原則——如果兩人擁有共同的朋友，他們彼此成為朋友的可能性很高，從而形成一個“三元組”（三角關係）



----

## Marketing 

###### Influencer-brand collaboration network **帶貨達人-品牌合作網路中的三元閉包應用**
<style scoped>
section {
    font-size: 25px;
}
</style>
`Triadic closure` can predict brand partnerships: if two influencers collaborate with the same brand, they may co-promote. `三元閉包`可用於預測品牌合作：如果兩位帶貨達人曾與同一品牌合作，他們未來很可能共同推廣該品牌。

**Influencer-to-Influencer** predictions**帶貨達人與帶貨達人之間的合作預測**:
<style scoped>
section {
    font-size: 25px;
}
</style>
- Alice and Bob share **Nike** → high chance they co-promote (e.g., joint Nike campaign) Alice 和 Bob 都曾與 **Nike** 合作 → 他們共同推廣（例如聯合參與 Nike 行銷活動）的可能性很高.
- Alice and Charlie share **Adidas** → likely to collaborate on Adidas content Alice 和 Charlie 都曾與 **Adidas** 合作 → 很可能共同創作 Adidas 相關內容.
- Bob and Charlie share **CocaCola** → strong candidate for a joint sponsored post Bob 和 Charlie 都曾與 **Coca-Cola** 合作 → 是聯合發佈贊助內容的有力候選組合.

----

**Brand-to-Brand** predictions**品牌與品牌之間的合作預測**:
<style scoped>
section {
    font-size: 25px;
}
</style>
- Nike and Adidas share **Alice** → potential for co-branded event (rare but possible in special campaigns) **Nike** 和 **Adidas** 都曾與 Alice 合作 → 有可能舉辦聯名活動（雖較罕見，但在特殊行銷活動中可能出現）.
- Nike and CocaCola share **Bob** → good opportunity for cross-promotion **Nike** 和 **Coca-Cola** 都曾與 Bob 合作 → 具備跨品牌聯合推廣的良好契機.



----

## Finance

###### Co-investment network **共同投資網路中的三元閉包應用**
<style scoped>
section {
    font-size: 25px;
}
</style>
In investor networks, `triadic closure` predicts new co-investments: if two investors back the same startup, they may invest together in another. 在投資人網路中，`三元閉包`可用於預測新的共同投資行為：如果兩位投資人曾投資同一家初創企業，他們未來很可能在其他專案中再次聯合投資。

**Investor-to-Investor** predictions (most relevant for co-investment) 投資人與投資人之間的合作預測（對共同投資最具參考價值）:

- InvestorA and InvestorB both backed **StartupX** → high likelihood they will co-invest in a new deal together InvestorA 和 InvestorB 都曾投資 **StartupX** → 他們未來在新項目中聯合投資的可能性很高。.
- InvestorA and InvestorC both backed **StartupY** → good candidates for syndication in future rounds. InvestorA 和 InvestorC 都曾投資 **StartupY** → 是未來融資輪次中組成投資團（syndication）的優質候選組合。

----
<style scoped>
section {
    font-size: 25px;
}
</style>
- InvestorB and InvestorC both backed **StartupZ** → strong signal for future joint investment. InvestorB 和 InvestorC 都曾投資 **StartupZ** → 強烈預示未來可能進行聯合投資
- InvestorA/B/D all share **StartupX** → InvestorA and InvestorD, or InvestorB and InvestorD are also likely to team up. InvestorA、InvestorB 和 InvestorD 都曾共同投資 **StartupX** → InvestorA 與 InvestorD，或 InvestorB 與 InvestorD 之間也極有可能在未來展開合作

**Startup-to-Startup** predictions **初創企業與初創企業之間的關聯預測**:

- StartupX and StartupY share **InvestorA** → they may attract overlapping investor interest in the future**StartupX** 和 **StartupY** 擁有共同投資人 **InvestorA** → 它們未來可能吸引重疊的投資人關注，形成相似的投資組合偏好



----

## Network Relation II: Structural Holes
<style scoped>
section {
    font-size: 25px;
}
</style>
**Structural holes** (Ronald Burt's theory) occur when a node bridges disconnected groups, providing brokerage advantages (e.g., information control). Measured by low **constraint** (high structural holes mean low redundancy in ties) 根據羅奈爾得·伯特（Ronald Burt）的結構洞理論，當某個節點（如行銷者或品牌）連接了原本互不相連的消費者群體時，該節點便佔據了“結構洞”位置。這種位置通過**低約束度**（low constraint）——即其社交聯繫缺乏冗餘，能接觸多元、非重疊的資訊源. 


---
## Social Media Analysis
<style scoped>
section {
    font-size: 25px;
}
</style>
Nodes with `low constraint (e.g., bridges)` control information flow but risk disconnection if removed. 約束度低的節點（如橋樑型節點）在資訊流動中扮演關鍵角色，但一旦被移除，可能導致網路斷裂

- **Node 2**: constraint = **0.367** → **lowest** (strongest broker)
- **Node 5**: constraint = **0.367** → **lowest** (strongest broker)

These two nodes have **significantly lower constraint** than the others (0.367 vs. 0.583).這兩個節點的約束度**顯著低於其他節點**（0.367 vs. 0.583），表明它們是連接不同社群的關鍵橋樑。



----
<style scoped>
section {
    font-size: 25px;
}
</style>
Nodes with **low constraint** (2 and 5) are **brokers** controlling information/flow between clusters. 

**Benefit**: They have unique access and power (structural hole advantage) 節點2和5作為**仲介者**（brokers），掌控著不同聚類之間的資訊、資源或機會流動，享有獨特的資訊優勢和影響力（即“結構洞紅利”）.

**Risk**: If **both** brokers are removed (e.g., they leave the company, stop investing, get banned from a platform), the network **fragments** into isolated groups 如果**這兩個仲介者同時被移除**（例如離職、停止投資、被平臺封禁等），整個網路將**分裂為彼此隔離的子群**. 

- Information, resources, or opportunities no longer flow between the two clusters 不同群體之間無法再傳遞資訊、資源或合作機會; The whole network loses cohesion and efficiency網路整體的連通性、協作效率和韌性大幅下降.

If there is **redundancy** (multiple bridges), removing one broker has little impact — the other maintains connectivity. 如果網路中存在**冗餘橋樑**（即多個節點同時連接相同群體），那麼移除單一仲介者的影響較小——其他橋樑仍可維持連通性。但在當前情境中，僅節點2和5具備低約束特徵，因此其角色尤為關鍵且脆弱。



---

## Marketing

##### Brand-consumer network
<style scoped>
section {
    font-size: 25px;
}
</style>
Marketers bridging consumer segments (structural holes) can introduce products to new groups efficiently. **佔據結構洞的行銷者**（如KOL、社群運營者或品牌本身）能夠充當不同消費者群體之間的橋樑。

- 他們可將產品或資訊從一個已滲透的群體高效傳遞至另一個尚未觸達的群體，從而**加速市場擴散**、**降低獲客成本**。
- 這種仲介角色賦予其獨特的**資訊控制權與影響力**：例如，率先向新圈層推薦新品、塑造跨群體口碑，甚至引導消費趨勢。

因此，`識別`並 `賦能`處於 結構洞 位置的節點，是品牌實現跨圈層傳播和精准破圈的關鍵策略。



---

## Finance

##### Interbank lending network
<style scoped>
section {
    font-size: 25px;
}
</style>
Banks with structural holes access diverse funding sources, reducing risk during crises.



---

## Network Relation III: Weak tie theory
<style scoped>
section {
    font-size: 25px;
}
</style>
**Weak tie theory** (Mark Granovetter's "Strength of Weak Ties") posits that weak ties (bridges between groups) are crucial for novel information diffusion, **弱連接**（如點頭之交、跨圈層的熟人關係）是**新穎資訊傳播的關鍵管道**while strong ties reinforce within groups 而**強連接**（如密友、家人）則主要在群體內部強化已有資訊與情感支援. 



`Weak ties` often coincide with bridge edges (high betweenness, removal increases components) 弱連接通常表現為**橋接不同社群的邊**（bridge edges），具有以下特徵. 

- **高仲介中心性**（high betweenness）：位於多條最短路徑上；**移除後會增加網路連通分量數量**，導致社群割裂；是**跨群體病毒式傳播**（viral spread）的核心驅動力。

----
<style scoped>
section {
    font-size: 25px;
}
</style>
Weak ties enable `viral spread` across communities

弱連接的雙重特性

- **脆弱性**：弱連接本身信任度較低、互動頻率不高，在局部關係中顯得“弱”；
- **戰略性價值**：在全球或網路層面卻極為“強”——它們將分散的社會結構聯結成整體，是`創新擴散`、`職業流動`、`市場滲透`和`資訊多樣性`的主要引擎。

Removing weak ties fragments the network, slowing diffusion 若移除弱連接，網路將**碎片化**，資訊傳播速度顯著下降; They bring diverse info but less trust than strong ties 儘管弱連接缺乏深度信任，但其帶來的**異質性資訊**（如新工作機會、新興趨勢、跨界合作）往往是強連接無法提供的.

Weak ties are "weak" locally (not strong friendships) but "strong" globally — they provide the strength that holds diverse social structures together and drives innovation, job mobility, and diffusion. 因此，弱連接雖“弱”，卻是社會網路中推動變革與連接多元世界的真正支柱. 



----

## Marketing
<style scoped>
section {
    font-size: 25px;
}
</style>
##### `Viral marketing/seeding` 病毒式行銷與種子使用者策略

`Seed` influencers with weak ties for broad reach in campaigns 在病毒式行銷（viral marketing）中，**種子用戶的選取策略**直接決定傳播的廣度與深度. 

- Strong ties (dense communities) spread messages deeply **within a group**. **強連接型影響者**（Strong-tie influencers）： 通常活躍于高密度、緊密聯繫的社群內部（如垂直興趣圈、地域社群）。 → 優勢在於能將資訊**深度滲透**到特定群體，引發高度信任與互動，但傳播範圍有限。

- **Weak ties** (high-betweenness influencers) spread messages **broadly across groups**. **弱連接型影響者**（Weak-tie influencers）： 擁有**高仲介中心性**（high betweenness），連接不同人群、興趣圈或人口統計群體（如跨圈層KOC、多領域內容創作者）。 → 雖然單個互動可能較淺、粉絲量未必龐大，卻能將資訊**廣泛擴散**至多個原本隔離的社群，觸發跨圈層傳播。

---
<style scoped>
section {
    font-size: 25px;
}
</style>
For **maximum reach and virality** 為實現**最大覆蓋範圍與病毒潛力**，應優先投放那些, prioritize seeding influencers who connect different demographics, interests, or niches — even if they have fewer followers than "big" in-group influencers.  連接**不同人口特徵、興趣領域或文化圈層**的影響者；即使其粉絲數少於某圈層內的“頭部大V”，但因其佔據**結構洞**或擁有**弱連接橋樑作用**，能更高效地打破資訊孤島。

**關鍵洞察**：真正的病毒傳播不只依賴“聲量”，更依賴“連接多樣性”。弱連接是資訊從一個社群躍遷到另一個社群的跳板——播種于此，方能引爆全域。



---

## Finance

##### `Information diffusion` in trading networks 交易網路中的資訊擴散
<style scoped>
section {
    font-size: 23px;
}
</style>
Weak ties spread market rumors/news quickly across investor groups, impacting prices 在交易與投資網路中，**弱連接**（weak ties）在市場訊息（尤其是非公開或未經證實的消息，如謠言、早期新聞或情緒信號）的快速傳播中扮演關鍵角色

- **強連接**（如緊密合作的基金經理、同機構交易員）傾向於在小圈子內反復交換相似資訊，強化共識，但難以突破圈層
- **弱連接**（如跨機構熟人、行業會議結識的連絡人、社交媒體上的泛關注關係）則充當**不同投資者群體之間的橋樑**，使市場傳聞或突發新聞能**迅速跨越社群邊界**

市場影響機制: 一條未經驗證的“利好”或“利空”消息，若通過弱連接從一個投資者群傳至另一個原本隔離的群體，可能**觸發連鎖反應**（如跟風買入/拋售）；這種跨群體的快速擴散會**放大市場情緒**，導致資產價格在缺乏基本面支撐的情況下劇烈波動；尤其在高頻交易或散戶高度活躍的市場中，弱連接驅動的資訊級聯（information cascade）可顯著**加速價格調整**，甚至引發短期泡沫或踩踏。

啟示: 監管者需警惕弱連接網路中**謠言的病毒式傳播**；投資者應意識到：看似“邊緣”的資訊源（如某跨圈層KOL、非核心連絡人）可能正是市場情緒轉折的早期信號；從網路視角看，市場的效率不僅來自資訊透明，也受制於誰和誰相連——弱連接既是資訊流動的動脈，也是雜訊放大的通道。

----