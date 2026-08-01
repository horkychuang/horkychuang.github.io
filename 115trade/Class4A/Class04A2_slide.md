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

## Heckscher-Ohlin Model III

**國企 Wen-Bin Chuang**
**2026-09-14**

![bg right fit](./images/02201.jpg)

----

## Factor Price Equalization (FPE) Theorem

<style scoped>
section {
    font-size: 27px;
}
</style>

Factor Price Equalization (FPE) Theorem (developed by Paul Samuelson, 1948) argued that **free trade in goods alone** (without international factor mobility) will lead to complete equalization of **real factor prices** (wages and capital rents) across countries, even though factors themselves cannot move.

**Core Idea**: free trade in goods can substitute for international factor mobility. In simple terms: `Trade in goods acts as a substitute for trade in factors`. However, because many strong assumptions are required, FPE rarely holds fully in practice. 

---

#### Core Assumptions (Same as H-O Model)

<style scoped>
section {
    font-size: 25px;
}
</style>

| Assumption                            | Description                                      |
| ------------------------------------- | ------------------------------------------------ |
| Two countries, Two goods, Two factors | Standard 2×2×2 model                             |
| `Perfect Competition`                 | In both goods and factor markets                 |
| Constant Returns to Scale (CRS)       | Production functions are homogeneous of degree 1 |
| Identical Homothetic Preferences      | Same tastes across countries                     |
| Technology                            | Same technology in both countries                |
| Factors mobile within country         | Labor and Capital mobile between sectors         |
| Factors immobile between countries    | No international factor mobility                 |
| No transport costs / trade barriers   | Free trade                                       |
| Full employment                       | All factors are fully utilized                   |

----

#### Main Mechanism 

<style scoped>
section {
    font-size: 23px;
}
</style>

1. Before trade (Autarky):

   - Labor-abundant country has **low wage (w)** and **high rental rate (r)**.
   - Capital-abundant country has **high wage (w)** and **low rental rate (r)**.

2. When trade opens:

   - Labor-abundant country exports labor-intensive good → increases demand for labor → **w rises**, **r falls**.
   - Capital-abundant country exports capital-intensive good → increases demand for capital → **r rises**, **w falls**.

3. Goods prices equalize across countries due to free trade.

4. Because technology is identical and goods prices are equal, cost minimization conditions

   force:
   $$
   \frac{w}{r} = \frac{w^*}{r^*}
   $$
   → `Real factor prices (w and r)` become equal  in both countries.

---

<style scoped>
section {
    font-size: 27px;
}
</style>

1. Before trade: Different relative endowments → different relative factor prices ($w/r$ differs).

2. Trade opens → countries export goods using their abundant factor → relative goods prices converge.
3. Goods price equalization → identical zero-profit lines in both countries.
4. Since the unit input requirements ($a_{ij}$) depend on factor prices, the only solution consistent with both equations is **equal factor prices**

**Key Insight**: FPE is a **price transmission result**. Trade equalizes output prices; technology links output prices to input prices; thus, input prices equalize.

----

#### Important Implication

<style scoped>
section {
    font-size: 27px;
}
</style>

1. **Benchmark, Not Prediction**: FPE serves as a theoretical baseline to diagnose why convergence fails (e.g., tech gaps, trade frictions, institutional barriers).
2. **Regional Integration**: FPE holds more strongly within integrated markets (EU single market, US interstate trade, NAFTA/USMCA supply chains).
3. **Development Strategy**: Highlights that trade alone cannot equalize living standards without technology transfer, human capital investment, and institutional reform.
4. **Global Value Chains**: Task-based trade allows partial factor price convergence for specific skills/tasks, even if absolute wages diverge.
5. **Migration vs. Trade Policy**: If FPE held perfectly, migration restrictions would be unnecessary. Its failure explains why labor mobility remains politically and economically relevant.

---

#### Key Takeaways

<style scoped>
section {
    font-size: 27px;
}
</style>

1. FPE is a **logical consequence** of the H-O assumptions, not an empirical law.
2. It relies critically on **incomplete specialization** and the **diversification cone**.
3. The **magnification effect** (SS) and **output reallocation** (Rybczynski) are the adjustment mechanisms that enable FPE.
4. Empirical failure of FPE reveals the importance of technology, institutions, trade costs, and factor heterogeneity.
5. Modern trade theory uses FPE as a **counterfactual benchmark** to measure frictions, assess convergence, and design integration policies.

----

#### Link to Previous Theorems

<style scoped>
section {
    font-size: 27px;
}
</style>

| Theorem                       | What Changes             | What is Equalized / Affected       | Direction             |
| ----------------------------- | ------------------------ | ---------------------------------- | --------------------- |
| Stolper-Samuelson             | Goods prices             | Factor prices (magnified)          | Opposite directions   |
| Rybczynski                    | Factor endowments        | Outputs (magnified)                | Opposite directions   |
| **Factor Price Equalization** | Goods prices (via trade) | **Factor prices across countries** | Complete equalization |

- FPE is the **international counterpart** of Stolper-Samuelson: once goods prices are equalized, factor prices must also equalize.
- It explains why trade can substitute for factor mobility.

---

#### Why Factor-Price Equalization Often Fails in Reality

<style scoped>
section {
    font-size: 20px;
}
</style>

Here are the **main reasons** FPE does not hold in the real world:

| Reason                                    | Explanation                                            | Effect on FPE                     |
| ----------------------------------------- | ------------------------------------------------------ | --------------------------------- |
| **Different Technology**                  | Countries have different production functions          | FPE fails completely              |
| **Factor Intensity Reversal**             | Which good is labor-intensive changes with wage levels | FPE breaks down                   |
| **Transportation Costs & Trade Barriers** | Goods prices do not fully equalize                     | Partial or no equalization        |
| **More than Two Factors**                 | Skilled labor, unskilled labor, land, etc.             | Makes equalization very difficult |
| **Specialization (Outside Cone)**         | Countries completely specialize                        | FPE does not occur                |
| **Non-homothetic Preferences**            | Different demand patterns                              | Affects goods prices              |
| **Imperfect Competition**                 | Monopolies, unions, minimum wages                      | Prevents factor price adjustment  |
| **Political & Institutional Factors**     | Labor laws, unions, immigration restrictions           | Blocks wage convergence           |
| **Capital Mobility**                      | Capital moves easily, labor does not                   | Partial equalization only         |

----

#### Mathematical Derivation

###### 1. Zero-Profit Conditions

<style scoped>
section {
    font-size: 27px;
}
</style>

$$
P_X=a_{L_X}w+a_{K_X}r,\qquad P_Y=a_{L_Y}w+a_{K_Y}r
$$

With `CRS` and `cost minimization`, $a_{i_j}=a_{i_j}(w/r)$ . At constant goods prices, $w/r$ is pinned down.

----

###### 2. Invertibility Condition

<style scoped>
section {
    font-size: 27px;
}
</style>

Write in matrix form:
$$
\left[\begin{array}{c}
P_{X}\\
P_{Y}
\end{array}\right]=\left[\begin{array}{cc}
a_{L_{X}} & a_{K_{X}}\\
a_{L_{Y}} & a_{K_{Y}}
\end{array}\right]\left[\begin{array}{c}
w\\
r
\end{array}\right]
$$


The mapping from ($P_X,P_Y$) to (w,r) is `invertible` iff the determinant is non-zero:
$$
Δ=a_{L_X}a_{K_Y}−a_{L_Y}a_{K_X}≠0
$$


This holds exactly when **no factor intensity reversal** occurs (i.e., X is consistently more/less labor-intensive than Yacross all $w/r$).

###### 3. Equalization Result

<style scoped>
section {
    font-size: 27px;
}
</style>

If $P_X^H=P_X^F$  and $P_Y^H=P_Y^F$ , and `technologies are identical`, then:
$$
\left[\begin{array}{c}
w_{H}\\
r_{H}
\end{array}\right]=A^{-1}\left[\begin{array}{c}
P_{X}\\
P_{Y}
\end{array}\right]=\left[\begin{array}{c}
w_{F}\\
r_{F}
\end{array}\right]
$$
**Absolute FPE** holds. Relative factor price equality $(w/r)_H=(w/r)_F$ follows immediately.

------

<style scoped>
section {
    font-size: 27px;
}
</style>

Under `free trade`, identical production technologies, and incomplete specialization, relative and absolute factor prices will equalize across countries, even if factors are internationally immobile.



In the 2×2×2 framework: If Home and Foreign trade freely in X and Y, and both produce both goods, then:

$$
w_H=w_F,\quad r_H=r_F
$$

- Trade in goods ≡≡ implicit trade in factor services.

----

###### 1. Zero-Profit Conditions (Same in Both Countries)

<style scoped>
section {
    font-size: 27px;
}
</style>

Because technologies are identical and markets are competitive:

**Home:**
$$
\begin{aligned} w_H a_{LX} + r_H a_{KX} &= P_X \\ w_H a_{LY} + r_H a_{KY} &= P_Y \end{aligned}
$$


**Foreign:**
$$
\begin{aligned} w_F a_{LX} + r_F a_{KX} &= P_X \\ w_F a_{LY} + r_F a_{KY} &= P_Y \end{aligned}
$$
Since `free trade` equalizes goods prices ($P_X$ and $P_Y$ are the **same** in both countries), both countries face the **identical system** of two equations with two unknowns ($w, r$).

----

###### 2. Matrix Form

<style scoped>
section {
    font-size: 25px;
}
</style>

$$
\mathbf{A}' \begin{bmatrix} w \\ r \end{bmatrix} = \begin{bmatrix} P_X \\ P_Y \end{bmatrix}
$$

Because the `input coefficient matrix` $\mathbf{A}$ is the same in both countries (`identical technology` + `same goods prices` → `same factor prices` from cost minimization), it follows that:
$$
\begin{bmatrix} w_H \\ r_H \end{bmatrix} = \begin{bmatrix} w_F \\ r_F \end{bmatrix}
$$


or `in percentage change` form (when goods prices change):
$$
\begin{bmatrix} \hat{w} \\ \hat{r} \end{bmatrix} = (\boldsymbol{\Theta})^{-1} \begin{bmatrix} \hat{P}_X \\ \hat{P}_Y \end{bmatrix}
$$

The right-hand side is identical for both countries.

----

###### Conditions Required for FPE

<style scoped>
section {
    font-size: 27px;
}
</style>

1. **Identical technologies** across countries.
2. **No factor intensity reversals** (factor intensity ranking remains the same at all factor prices).
3. **Both countries produce both goods** (they remain in the **cone of diversification** — no complete specialization).
4. **Free trade** with no transport costs or trade barriers.
5. **Perfect competition** and constant returns to scale.
6. Same number of factors and goods (or at least as many goods as factors).

---

###### Conditions for FPE to Hold

<style scoped>
section {
    font-size: 27px;
}
</style>

**FPE holds only if**:

- Identical technology; No factor intensity reversal
- Both countries continue producing **both goods** (diversification)
- Free trade with no transport costs; Same homothetic preferences

In reality, **most of these assumptions are violated**, which is why we do **not** observe full equalization of wages and capital returns between rich and poor countries.

---

<style scoped>
section {
    font-size: 27px;
}
</style>

Stolper-Samuelson and Rybczynski theorems examine `within-country` adjustments, FPE addresses `cross-country` convergence: it shows that, under strict conditions, free trade in goods can perfectly substitute for international factor mobility, equalizing wages and returns to capital across nations.

#### Historical Context & Core Statement

- **Origin**: Paul Samuelson (1948), *"International Trade and the Equalization of Factor Prices"* (Economic Journal).
- **Purpose**: To demonstrate the full general equilibrium implications of the H-O model when goods prices converge through trade.

---

## Leontief Paradox

<style scoped>
section {
    font-size: 25px;
}
</style>

The **Leontief Paradox（1953)** is one of the most famous empirical challenges to the **Heckscher-Ohlin (H-O) model**. Wassily Leontief tested the H-O theorem using 1947 U.S. trade data. According to H-O, the United States — the most **capital-abundant** country in the world — should export **capital-intensive** goods and import **labor-intensive** goods. **Empirical Result**:

- U.S. **exports** were more **labor-intensive** than U.S. **imports**.
- Specifically, U.S. imports were about **30% more capital-intensive** than U.S. exports.

**Core Idea**: It showed that reality is more complex than “countries export goods that intensively use their abundant factors.” It opened the door to richer theories that include `human capital`, `technology gaps`, and `increasing returns to scale`.  The empirical result was a **direct contradiction** of the H-O prediction → hence called a “paradox.”

----

#### Key Empirical Studies

<style scoped>
section {
    font-size: 22px;
}
</style>

| Year   | Researcher                  | Country      | Main Finding                                  | Supports H-O? |
| ------ | --------------------------- | ------------ | --------------------------------------------- | ------------- |
| 1953   | Wassily Leontief            | USA (1947)   | Exports more labor-intensive than imports     | **No**        |
| 1971   | Robert Baldwin              | USA (1962)   | Imports 27% more capital-intensive            | **No**        |
| 1980   | Edward Leamer               | USA          | Paradox disappears with better methodology    | Mixed         |
| 1987   | Bowen, Leamer & Sveikauskas | 27 countries | Mixed results; H-O rejected in many cases     | Weak          |
| 2005   | Kwok & Yu                   | USA          | Paradox reduced or disappears in updated data | Improved      |
| Recent | Various                     | USA          | Paradox still appears, especially in services | Mixed         |

---

#### Main Explanations for the Leontief Paradox

<style scoped>
section {
    font-size: 27px;
}
</style>

Several attempts have been made to resolve the paradox:

1. `Higher U.S. Labor Productivity`  (Leontief’s own explanation)
   - American workers were more productive (better technology, education, management).
   - When adjusted for **productivity-equivalent labor**, the paradox disappears.
2. `Factor Intensity Reversal`
   - The assumption that a good is always labor- or capital-intensive may not hold across countries.
3. `Human Capital / Skilled Labor`
   - U.S. exports are intensive in **skilled labor** (human capital), not raw labor. Standard H-O only considers unskilled labor and physical capital.

----

<style scoped>
section {
    font-size: 27px;
}
</style>

4. `Natural Resources`
   - Many U.S. imports are resource-intensive (oil, minerals), which are capital-intensive to extract.
5. `Methodological Issues`
   - Leontief compared gross exports and imports instead of net trade.
   - Did not account for tariffs and trade barriers.
6. `Technology Differences`
   - H-O assumes identical technology across countries, which is unrealistic.

----

#### Current Status (Modern View)

<style scoped>
section {
    font-size: 27px;
}
</style>

- The strict version of the H-O model is **not strongly supported** by empirical evidence.
- The Leontief Paradox **still appears** in many studies, especially for the United States.
- However, when researchers include **human capital**, **technology differences**, or use **value-added trade** data, the paradox is **significantly reduced** or disappears in many cases.
- Modern trade theory has moved beyond simple H-O toward models that incorporate **technology**, **economies of scale**, **product differentiation**, and **institutions**.

