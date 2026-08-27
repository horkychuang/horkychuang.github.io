---
marp: true
theme: gaia
color: #000
size: 16:9
colorSecondary: #333
backgroundColor: #fef9e7;
backgroundImage: url('images/background_1.JPG')
footer: 'Designed by Wen-Bin 2026-09-05'
paginate: true
---

<!-- _class: lead -->

### Digital and Service Trade

#### 數位與服務貿易

**Wen-Bin Chuang**
**2026-09-14**

![bg right fit](./images/02201.jpg)



------

<style scoped>
section {
    font-size: 23px;
}
</style>

### Digital Trade

`Digital trade` includes digitally ordered goods/services and especially digitally delivered services (e.g., `software`, `cloud computing`, `streaming`, e-commerce, data flows, ICT-enabled services). It includes cross-border data flows. 

#### **Main Categories of Digital Trade**

| Category                         | Examples                                                     | Importance                  |
| -------------------------------- | ------------------------------------------------------------ | --------------------------- |
| **Digitally Delivered Services** | Software, cloud computing, streaming, telemedicine, online education | Fastest growing             |
| **Digital Products**             | e-books, music, software downloads, 3D printing files        | Fully digital               |
| **E-commerce**                   | Cross-border online sales (Amazon, Shopee, Temu)             | Goods + digital             |
| **Data Flows**                   | Movement of personal, commercial, and government data        | Backbone of digital economy |

----
#### **Key Economic Characteristics**

<style scoped>
section {
    font-size: 27px;
}
</style>

- **Near-zero marginal cost**: 
  - Once created, digital goods/services can be replicated and delivered at almost zero extra cost.
- **Network Effects** and **Platform Economies** (winner-takes-most dynamics).
- **Scalability**: Global reach with minimal physical infrastructure.


---

**Main Economic Effects**:

<style scoped>
section {
    font-size: 27px;
}
</style>

- **Trade Cost Reduction**: Digital delivery dramatically lowers trade costs compared to physical goods.
- **Services Trade Boom**: Traditionally non-tradable services (e.g., consulting, education, healthcare) become tradable.
- **Market Expansion for SMEs**: Small firms can export directly via platforms (e.g., Etsy, Shopify, Temu).
- **Data as a Factor of Production**: Similar to how capital and labor drive growth.

---
#### **Economic Models**

<style scoped>
section {
    font-size: 27px;
}
</style>

- **Endogenous Growth Theory**: Digital trade accelerates `idea flows` and innovation.
- **Gravity Model Extensions**: Digital trade reduces “distance” dramatically (time and cost).
- **Baumol’s Cost Disease**: Services that were non-tradable become tradable → productivity gains in services sector.

---
#### **Key Issues in Digital Trade**

<style scoped>
section {
    font-size: 27px;
}
</style>

- **Cross-border Data Flows** and localization requirements
- **Data Privacy** (GDPR in EU vs lighter rules elsewhere)
- **Cybersecurity and National Security**
- **Digital Taxation** (e.g., Digital Services Taxes)
- **Intellectual Property Protection** (piracy, counterfeiting)
- **Platform Regulation** (Big Tech dominance)

-----

#### **Major Agreements**

<style scoped>
section {
    font-size: 27px;
}
</style>

- **CPTPP** and **DEPA** (Digital Economy Partnership Agreement): High-standard digital rules
- **USMCA** (Chapter 19): Strong digital trade chapter
- **EU** approach: More regulatory (data protection first)
- **WTO** negotiations on e-commerce (ongoing)

----

### Service Trade

<style scoped>
section {
    font-size: 25px;
}
</style>

`Services trade`  (e.g., finance, tourism, transport, professional services, telecom) differs significantly from goods trade. It was traditionally harder to measure and model than goods trade due to its `intangibility and regulatory barriers`.  

- **Better fit than for goods**: Kimura and Lee (2006) found that gravity equations explain services trade better than goods trade. Their adjusted R² values were notably higher for services (around 0.80) compared to goods (around 0.65).

---
<style scoped>
section {
    font-size: 25px;
}
</style>
- **Core drivers**:
  - **Economic size (GDP)**: Strongly positive, as expected — larger economies trade more services.
  - **Common language**: One of the most robust and important positive factors (often stronger than for goods, due to the personal/cultural nature of many services).
  - **Geographic distance**: Frequently **insignificant or weaker** than in goods trade. This makes intuitive sense because many services (especially Mode 1 — cross-border supply via digital means) do not require physical transportation.
- **Other factors**: Cultural proximity, colonial ties, common legal systems, and trade agreements boost services flows. Regulatory barriers have a strong negative impact.

----

###### Basic Gravity Model (Traditional Goods Trade)

<style scoped>
section {
    font-size: 25px;
}
</style>

The `classic` gravity equation is:
$$
T_{ij} = G \frac{Y_i^\alpha Y_j^\beta}{D_{ij}^\theta}
$$


Where:

- $T_{ij}$ = Trade flow from country $i$ to country $j$
- $Y_i, Y_j$ = Economic size (GDP) of country $i$ and $j$
- $D_{ij}$ = Distance between $i$ and $j$,  $G$ = Constant
- $\alpha, \beta, \theta$ = Parameters to be estimated


In **log-linear form** (most commonly used for estimation):

$$
\ln(T_{ij}) = \alpha + \beta_1 \ln(Y_i) + \beta_2 \ln(Y_j) + \beta_3 \ln(D_{ij}) + \epsilon_{ij}
$$

----

###### Key Modified Gravity Equation for Services:

<style scoped>
section {
    font-size: 27px;
}
</style>

Services trade follows the gravity model **quite well**, but with some important differences compared to goods trade. 
$$
\ln(S_{ij}) = \alpha + \beta_1 \ln(Y_i) + \beta_2 \ln(Y_j) + \beta_3 \ln(D_{ij}) $$
$$
+ \beta_4 \text{Cultural}_{ij} + \beta_5 \text{Regulatory Similarity}_{ij} + \beta_6 \text{Mode Dummies} + \epsilon
$$

**Empirical Findings**:

- Distance elasticity ($\beta_3$) is typically **lower** for services than for goods (services are less sensitive to physical distance).
- **Cultural proximity** and **institutional quality** matter more for services.
-----

**Important Variables Added for Services**:

<style scoped>
section {
    font-size: 23px;
}
</style>

| Variable                            | Effect on Services Trade         | Reason                                                    |
| ----------------------------------- | -------------------------------- | --------------------------------------------------------- |
| **Distance ($D_{ij}$)**             | Negative (but weaker than goods) | Many services are delivered digitally or via Mode 3 (FDI) |
| **GDP / Market Size**               | Strongly Positive                | Larger economies trade more services                      |
| **Common Language / Colonial Ties** | Strongly Positive                | Very important for services                               |
| **Regulatory Similarity**           | Positive                         | Similar regulations reduce barriers                       |
| **Digital Infrastructure**          | Positive                         | Broadband, 5G, data centers                               |
| **Trade Restrictiveness (STRI)**    | Negative                         | Higher restrictions → lower trade                         |
| **Mode of Supply Dummies**          | Varies                           | Mode 1 & 3 usually larger                                 |

-----

#### Differences from Goods Trade

<style scoped>
section {
    font-size: 24px;
}
</style>

| Aspect               | Goods Trade                       | Services Trade                                               |
| -------------------- | --------------------------------- | ------------------------------------------------------------ |
| **Distance effect**  | Strong negative                   | Weaker or insignificant                                      |
| **Language/Culture** | Important                         | Often more critical                                          |
| **Main barriers**    | Tariffs, transport costs, borders | Regulations, licensing, data rules, FDI restrictions         |
| **Data quality**     | Better measured                   | Patchy, missing values, harder to observe                    |
| **Model fit**        | Good                              | Often better                                                 |
| **Modes of supply**  | Mainly physical                   | GATS Modes 1-4 (cross-border, consumption abroad, commercial presence, movement of persons) |

- Services trade involves more "dark costs" 
  - opaque regulations in sectors like professional services, finance, and telecom — compared to the more transparent tariffs on goods.



-----
#### **Key findings from studies**:

<style scoped>
section {
    font-size: 27px;
}
</style>

- `Economic size (GDPs)` remains the strongest positive driver.
- `Distance` has a significant negative effect (though sometimes weaker than for goods, due to lower physical transport needs).
- `Common language`, `cultural proximity`, and `institutional similarities` are particularly important.
- Services trade restrictions (measured by indices like the OECD `STRI`) strongly reduce flows.
- The model successfully explains why services trade is highly concentrated among high-income countries and is sensitive to regulatory barriers.

----

#### Services Trade Restrictiveness Index(STRI)

<style scoped>
section {
    font-size: 27px;
}
</style>

- The **STRI (經合組織服務貿易限制指數)**, developed by the OECD, is a groundbreaking tool because it quantifies what was historically unquantifiable: **non-tariff regulatory barriers in services**. Unlike goods, where tariffs are the primary barrier, services trade is restricted by domestic regulations (e.g., licensing, foreign equity limits, visa quotas)..

- To understand its impact on trade, we must first understand what STRI measures. It scores countries from **0 (completely open)** to **1 (completely closed)** across 22 service sectors based on five policy dimensions:

1. **Foreign Ownership Restrictions:** Limits on foreign equity, screening mechanisms, or bans on foreign land ownership.
2. **Movement of People:** Visa quotas, economic needs tests, and lack of mutual recognition of professional qualifications.

---
<style scoped>
section {
    font-size: 27px;
}
</style>
3. **Other Discriminatory Measures:** Subsidies only for domestic firms, public procurement restrictions, or forced technology transfer.
4. **Barriers to Competition:** State-owned monopolies, exclusive licenses, or limits on the number of operating firms.
5. **Regulatory Transparency:** Lack of public consultation, unclear licensing procedures, or lack of independent regulatory bodies.

-----

##### Integration with the Gravity Model

<style scoped>
section {
    font-size: 27px;
}
</style>

- Because services don't face `border tariffs`, economists use STRI to calculate the **Ad Valorem Equivalent (AVE)**
  - Essentially translating a regulatory restriction into a "tariff equivalent."

  - *Example:* An OECD study found that in some highly restricted sectors (like certain `telecommunications` or `professional services`), the STRI score translates to an AVE of **30% to over 100%**. This means the domestic regulation acts exactly like a 100% tariff, severely choking trade.
---
<style scoped>
section {
    font-size: 27px;
}
</style>
When applied to the Gravity Model, the equation looks like this:
$$
\ln(Trade_{ij}) = \beta_0 + \beta_1 \ln(GDP_i) + \beta_2 \ln(GDP_j) 
$$
$$
+ \beta_3 \ln(Distance_{ij}) - \beta_4 (STRI_i + STRI_j) + \epsilon
$$
**Empirical Findings:**

- **Elasticity:** A 10% reduction in a country's STRI score is empirically shown to increase its services imports by roughly **3% to 5%**, depending on the sector.
- **Symmetry:** The STRI of `both` the exporting and importing country matters. 
  - If the exporter has high STRI (e.g., poor regulatory transparency), it raises costs for their firms, reducing their export competitiveness.

- Perhaps the most critical insight regarding STRI is that services restrictions do not just hurt services trade; they hurt manufacturing and the broader economy.

------

<style scoped>
section {
    font-size: 27px;
}
</style>

- Interestingly, `Digital trade` often shows strong gravity patterns but with some nuances: 
  - `Distance` matters less for purely digital delivery, while `regulatory alignment` (data privacy, cybersecurity, digital standards) and `digital readiness` become critical "distance-like" frictions. 
  - That is to say, `Digital trade` (especially digitally delivered services) further weakens the role of physical distance.

- It adapts the classic framework to analyze trade in digitally ordered goods and especially digitally deliverable services (DDS), such as software, cloud computing, financial services, professional services, streaming content, telecom, and data-related flows.

---
<style scoped>
section {
    font-size: 27px;
}
</style>
The core structure remains the same:

- **Positive drivers**: 
  - Economic sizes (GDPs), digital readiness/connectivity.
- **Negative drivers**: 
  - Various forms of "distance" — geographic, regulatory, data localization rules, etc.

----

###### Differences from Traditional Services and Goods Trade

<style scoped>
section {
    font-size: 23px;
}
</style>

| Aspect                | Goods Trade             | Traditional Services   | Digital Trade (DDS)                      |
| --------------------- | ----------------------- | ---------------------- | ---------------------------------------- |
| **Distance**          | Strong negative         | Weaker                 | Weakest (but regulatory distance strong) |
| **Key enablers**      | Infrastructure, tariffs | Regulations, proximity | Digital connectivity, data rules         |
| **Policy focus**      | Tariffs, NTBs           | STRI, licensing        | E-commerce chapters, data flows          |
| **Complementarities** | With services           | With goods             | Strong with ICT goods & other services   |
| **Data challenges**   | Good                    | Patchy                 | Emerging but improving                   |

- Digital trade shows **strong complementarities**: Better digital infrastructure and ICT goods imports amplify services exports.

----

###### Extended Gravity Model for Digital Trade:

<style scoped>
section {
    font-size: 27px;
}
</style>

$$
\ln(DT_{ij,t}) = \alpha + \beta_1 \ln(GDP_{i,t}) + \beta_2 \ln(GDP_{j,t}) + \beta_3 \ln(Dist_{ij}) $$ 
$$+ \mathbf{X}_{ij,t}'\boldsymbol{\gamma} + \delta_{ij} + \lambda_t + \epsilon_{ij,t}
$$

Where:

- $DT_{ij,t}$ = `Digital trade flow` from country $i$ to $j$ in year $t$ (e.g., digitally delivered services, e-commerce, etc.)
- $GDP_{i,t}, GDP_{j,t}$ = Economic size (GDP)
- $Dist_{ij}$ = Geographical distance between countries $i$ and $j$
- $\mathbf{X}_{ij,t}$ = Vector of extended variables (most important part)
- $\delta_{ij}$ = Country-pair fixed effects, $\lambda_t$ = Year fixed effects, $\epsilon_{ij,t}$ = Error term

---

**Key Differences from Traditional Gravity Model**:

<style scoped>
section {
    font-size: 25px;
}
</style>

| Factor                       | Goods Trade     | Services Trade       | Digital Trade                      |
| ---------------------------- | --------------- | -------------------- | ---------------------------------- |
| **Distance Effect**          | Strong negative | Moderate negative    | **Weak / Sometimes insignificant** |
| **Border Effect**            | Strong          | Moderate             | Very weak                          |
| **Digital Connectivity**     | Minor           | Important            | **Very Strong**                    |
| **Data Policy**              | Not relevant    | Moderately important | **Critical**                       |
| **Language & Culture**       | Important       | Very Important       | Still important                    |
| **Regulatory Harmonization** | Important       | Very Important       | Extremely Important                |

---

#### The Digital Extension: DSTRI (Digital STRI)

<style scoped>
section {
    font-size: 25px;
}
</style>

As trade digitized, the OECD recognized that traditional STRI didn't fully capture `digital barriers`. They introduced the **Digital Services Trade Restrictiveness Index (DSTRI)**.

#### **New Digital Barriers Measured:**

1. **Data Localization:** Requirements that data generated domestically must be stored on domestic servers (massively increases costs for cloud and digital services).
2. **Source Code Disclosure:** Forcing foreign software companies to hand over proprietary source code as a condition of market access.
3. **Digital Signatures & E-contracts:** Lack of legal recognition for cross-border electronic contracts.

----

##### **Impact on Digital Trade:**

<style scoped>
section {
    font-size: 25px;
}
</style>

Using the Gravity Model with DSTRI, researchers found that data localization requirements alone can reduce digital trade flows by **up to 10-15%**. DSTRI is now crucial for understanding the trade of "digitally deliverable services" (e.g., software, streaming, cloud computing).