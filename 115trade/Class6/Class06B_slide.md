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

## New Trade Theory II

**Wen-Bin Chuang**
**2026-09-14**

![bg right fit](./images/02201.jpg)

-----

<style scoped>
section {    
    font-size: 27px;}
</style>

The **Melitz model (2003)** is the `most influential extension` of New Trade Theory. It introduces **firm heterogeneity** in `productivity` and explains why only some firms export, why exporters are larger and more productive, and how trade causes important reallocation effects within industries.



Core Idea: Within this framework, the **self-selection effect** is one of the most important and empirically robust concepts. **Only the most productive firms become exporters**

----

#### 1. Key Assumptions

<style scoped>
section {
    font-size: 27px;
}
</style>

- `Monopolistic competition` + `differentiated` varieties (like Krugman)
- Firms differ in **productivity** ($\phi$) — drawn from a Pareto distribution
- Increasing returns to scale (fixed cost + constant marginal cost)
- Two types of fixed costs:
  - Domestic fixed cost: $f_d$ and Export fixed cost: $f_x$ (higher than $f_d$)
- `Iceberg trade cost`:$\tau >1$ (to deliver 1 unit abroad, $\tau$ units must be shipped)
- `CES preferences` with elasticity of substitution $\sigma > 1$

-----

#### 2. Core Structure of the Model

###### Firm Revenue and Profit

<style scoped>
section {
    font-size: 25px;
}
</style>

A firm with productivity $\phi$ has:

- **Marginal cost**: $\frac{w}{\phi}$,  **Domestic price**: $p_d(\phi) = \left( \frac{\sigma}{\sigma-1} \right) \frac{w}{\phi}$
- **Export price** (in foreign market): $p_x(\phi) = \left( \frac{\sigma}{\sigma-1} \right) \frac{\tau w}{\phi}$

**Revenue**:
$$
r_d(\phi) = R \left( \frac{p_d(\phi)}{P} \right)^{1-\sigma}\\
r_x(\phi) = R^* \left( \frac{p_x(\phi)}{P^*} \right)^{1-\sigma}
$$


Where $R$ = total domestic expenditure,$P$ = price index.

then `profit`

$$
\pi_d(\phi) = \frac{r_d(\phi)}{\sigma} - w f_d, \quad
\pi_x(\phi) = \frac{r_x(\phi)}{\sigma} - w f_x
$$

----
###### 3. Key Cutoffs (Most Important Equations)

<style scoped>
section {
    font-size: 25px;
}
</style>

Firms decide whether to produce and export based on `productivity`:

- **Domestic production cutoff** ($\phi_d^*$): `Minimum productivity` to serve the domestic market $\pi_d(\phi_d^*) = 0 \quad \Rightarrow \quad r_d(\phi_d^*) = \sigma w f_d$
- **Export cutoff** ($\phi_x^*$): `Minimum productivity` to export $\pi_x(\phi_x^*) = 0 \quad \Rightarrow \quad r_x(\phi_x^*) = \sigma w f_x$

Moreover, their profit is 

- $\pi_d(\phi)=wf_d\left[\left(\frac{\phi}{\phi_d^*}\right)^{\sigma-1}-1\right]$ and $\pi_x(\phi)=wf_x\left[\left(\frac{\phi}{\phi_x^*}\right)^{\sigma-1}-1\right]$



Because of `trade costs` ($\tau$) and `higher fixed export cost` ($f_x$), we get:
$$
\phi_x^* > \phi_d^*
$$

**Only the most productive firms export.**

----

##### Summary of Key Results

<style scoped>
section {
    font-size: 25px;
}
</style>

1. **Revenue increases with productivity:** $r(\phi)\propto \phi^{\sigma−1}$. `More productive` firms sell more because they charge `lower prices`.
2. **Profit is convex in productivity:** $\pi(\phi)\propto \phi^{\sigma−1}$ **constant**. Small differences in productivity lead to large differences in profit.
3. **Selection effect:** Only firms with $\phi≥\phi_d^∗$  survive domestically. Only firms with $\phi≥\phi_x^∗$  export. Since $\phi_x^∗>\phi_d^∗$ , the most productive firms export, while less productive firms serve only the domestic market, and the least productive firms `exit`.
4. **Reallocation effect:** `Trade liberalization` (lower $\tau$) raises $\phi_d^∗$ , forcing the least productive firms to `exit`. Resources are reallocated toward `more productive firms`, raising aggregate productivity. This is the core mechanism through which trade generates gains in the Melitz model.

----

###### 4. Productivity Distribution (Pareto)

<style scoped>
section {
    font-size: 27px;
}
</style>

Productivity $\phi$ is drawn from a Pareto distribution:
$$
G(\phi) = 1 - \left( \frac{\phi_{min}}{\phi} \right)^k \quad \text{for } \phi \geq \phi_{min}
$$


Where $k > \sigma - 1$ (shape parameter). This assumption delivers very nice closed-form solutions.

----

##### 5. Main Predictions of the Melitz Model

<style scoped>
section {
    font-size: 25px;
}
</style>

| Prediction                      | Explanation                                                  |
| ------------------------------- | ------------------------------------------------------------ |
| **Selection into Exporting**    | Only firms with $\phi > \phi_x^*$ export                     |
| **Exporter Premium**            | Exporters are larger, more productive, and pay higher wages  |
| **Intra-industry Reallocation** | Trade liberalization → low-productivity firms exit, resources move to more productive firms |
| **Productivity Gains**          | Aggregate industry productivity rises even without new technology |
| **Gains from Trade**            | Larger than in Krugman (includes reallocation gains)         |

-----

##### 6. Gains from Trade in Melitz Model

<style scoped>
section {
    font-size: 25px;
}
</style>

Trade generates **three types of gains**:

1. **Variety gains** (same as Krugman)
2. **Pro-competitive gains** (lower markups due to more competition)
3. **Reallocation / Selection gains** — most important new channel

When trade opens:

- Domestic cutoff $\phi_d^*$ rises → least productive firms exit
- Export cutoff $\phi_x^*$ falls → more firms start exporting
- Resources reallocate toward higher-productivity firms → average industry productivity increases

This `reallocation effect` can be `very large`.

---

##### 7. Key Equation: Aggregate Productivity

<style scoped>
section {
    font-size: 25px;
}
</style>

`Average industry productivity` ($\tilde{\phi}$) rises with trade openness. In equilibrium, the weighted average productivity is:
$$
\tilde{\phi} = \left[ \frac{1}{1-G(\phi_d^*)} \int_{\phi_d^*}^{\infty} \phi^{\sigma-1} dG(\phi) \right]^{\frac{1}{\sigma-1}}
$$
With Pareto distribution, this simplifies nicely and shows clear productivity gains from lower trade costs.

----

### Mathematical Derivation

###### 1. Firm Profit and Zero-Profit Conditions

###### Firm Revenue

<style scoped>
section {
    font-size: 25px;
}
</style>

A firm with productivity $\phi$ faces `CES demand`. Its `revenue` in the domestic market is:
$$
r_d(\phi) = R \left( \frac{p_d(\phi)}{P} \right)^{1-\sigma}
$$


where:

- $R$ = total domestic expenditure (income)
- $P$ = domestic price index;  $\sigma$= elasticity of substitution

----

<style scoped>
section {
    font-size: 25px;
}
</style>

The optimal price (markup pricing):
$$
p_d(\phi) = \left( \frac{\sigma}{\sigma-1} \right) \frac{w}{\phi} = \mu \frac{w}{\phi}
$$


Substitute price into revenue:
$$
r_d(\phi) = R \left( \frac{\mu w / \phi}{P} \right)^{1-\sigma} = R \left( \frac{\mu w}{P} \right)^{1-\sigma} \phi^{\sigma-1}
$$


Let $B = R \left( \frac{\mu w}{P} \right)^{1-\sigma}$ (a market demand index). Then:
$$
r_d(\phi) = B \cdot \phi^{\sigma-1}
$$

**Export revenue** (with iceberg trade cost $\tau$):


$$
r_x(\phi) = B^* \cdot (\phi / \tau)^{\sigma-1}
$$

-----

#### Operating Profit

<style scoped>
section {
    font-size: 27px;
}
</style>

Because of `CES demand`, operating profit (`revenue minus variable cost`) is revenue divided by $\sigma$:
$$
\pi_d^{op}(\phi) = \frac{r_d(\phi)}{\sigma} = \frac{B}{\sigma} \phi^{\sigma-1}, \quad
\pi_x^{op}(\phi) = \frac{r_x(\phi)}{\sigma} = \frac{B^*}{\sigma} \left( \frac{\phi}{\tau} \right)^{\sigma-1}
$$

----

#### Zero-Profit Conditions (Net Profit = 0)

<style scoped>
section {
    font-size: 27px;
}
</style>

**Domestic market**:
$$
\pi_d(\phi) = \frac{r_d(\phi)}{\sigma} - w f_d = 0, \quad
\frac{B}{\sigma} \phi_d^{*(\sigma-1)} = w f_d
$$


**Export market**:
$$
\frac{B^*}{\sigma} \left( \frac{\phi_x^*}{\tau} \right)^{\sigma-1} = w f_x
$$

----

<style scoped>
section {
    font-size: 27px;
}
</style>

Solving for the **productivity cutoffs**:
$$
\phi_d^* = \left( \frac{\sigma w f_d}{B} \right)^{\frac{1}{\sigma-1}},\quad
\phi_x^* = \tau \left( \frac{\sigma w f_x}{B^*} \right)^{\frac{1}{\sigma-1}}
$$


**Key Relationship**:
$$
\phi_x^* > \phi_d^* \quad \text{(because } f_x > f_d \text{ and } \tau > 1\text{)}
$$


- Only firms with productivity $\phi \geq \phi_d^*$ produce for the domestic market. 
- Firms with $\phi \geq \phi_x^*$ also export.

------

###### 2. Free Entry Condition

<style scoped>
section {
    font-size: 27px;
}
</style>

Firms pay a `sunk entry cost` $f_e$ before drawing their productivity $\phi$ from the Pareto distribution.

The **expected profit** must equal the entry cost in equilibrium:
$$
E[\pi] = \int_{\phi_d^*}^{\infty} \left[ \frac{B}{\sigma} \phi^{\sigma-1} - w f_d \right] dG(\phi) + \int_{\phi_x^*}^{\infty} \left[ \frac{B^*}{\sigma} \left( \frac{\phi}{\tau} \right)^{\sigma-1} - w f_x \right] dG(\phi) = w f_e
$$


With Pareto distribution, this equation closes the model and determines the cutoffs.

------

#### 3. Home Market Effect in Heterogeneous Firms Model

<style scoped>
section {
    font-size: 27px;
}
</style>

The **Home Market Effect (HME)** is stronger in the Melitz model than in the **original Krugman model**.

###### Setup

- Two countries: `Home (larger)` and `Foreign (smaller)`
- $L_H > L_F → R_H > R_F$ (larger domestic expenditure in Home)
- Identical technology, fixed costs, and trade costs

----

#### Mechanism

<style scoped>
section {
    font-size: 25px;
}
</style>

Because $R_H > R_F$, the domestic demand index $B_H > B_F$. From the cutoff equations:

$$
\phi_d^{H*} < \phi_d^{F*}
$$

**Results**:

1. **Lower domestic cutoff in large country**: More firms can survive in the large market.
2. **Lower export cutoff for firms in large country**: Firms from the large country find it easier to export ($\phi_x^{H*}$ is relatively lower).
3. **Disproportionate export share**: The larger country exports a **more than proportional** share of varieties relative to its size.

**Intuition**: The `large domestic market` allows firms to cover fixed costs more easily. This gives firms based in the large country a cost advantage (they operate at lower average cost due to scale), making them more competitive in the export market as well.

---

#### Mathematical Expression of HME

<style scoped>
section {
    font-size: 27px;
}
</style>

The share of firms from Home that export is higher, and the value of exports from Home to Foreign exceeds the size ratio:
$$
\frac{\text{Exports from H to F}}{\text{Exports from F to H}} > \frac{R_H}{R_F} > 1
$$
This is the strong form of the Home Market Effect in the Melitz model.

------

##### Summary of Key Insights

<style scoped>
section {
    font-size: 27px;
}
</style>

- **Productivity cutoffs** ($\phi_d^*$ and $\phi_x^*$) are `endogenous` and respond to market size, trade costs, and fixed costs.
- `Trade liberalization` lowers $\phi_x^*$ (more exporters) and raises $\phi_d^*$ (exit of least productive firms) → **reallocation toward more productive firms**.
- The **Home Market Effect** is `amplified`: larger countries host more firms and have a disproportionately large export share.

---

##### Definition of Aggregate Productivity

<style scoped>
section {
    font-size: 25px;
}
</style>

In the Melitz model, **aggregate (industry) productivity** $\tilde{\phi}$ is defined as the **weighted average productivity** that enters the price index and welfare calculations. It is given by:
$$
\tilde{\phi} = \left[ \frac{1}{M} \int_{\phi_d^*}^{\infty} \phi^{\sigma-1} \, dM(\phi) \right]^{\frac{1}{\sigma-1}}
$$


where:

-  M = mass (number) of producing firms;  $\phi_d^*$ = domestic productivity cutoff
-  $\sigma$= elasticity of substitution

This is the CES-weighted average productivity. Higher $\tilde{\phi}$ means lower industry price index and higher welfare.

------

###### Productivity Distribution

<style scoped>
section {
    font-size: 27px;
}
</style>

Firms draw productivity $\phi$ from a **Pareto distribution**:
$$
G(\phi) = 1 - \left( \frac{\phi_{min}}{\phi} \right)^k \quad \text{for } \phi \geq \phi_{min}, \quad
g(\phi) = k \phi_{min}^k \phi^{-(k+1)}
$$


where $k > \sigma - 1$ (shape parameter — higher  k means less dispersion). The conditional distribution of productivity among **active firms** (those with $\phi \geq \phi_d^*$) has a nice truncated Pareto form.

------

##### Derivation of Aggregate Productivity

<style scoped>
section {
    font-size: 27px;
}
</style>

Let $M_d$ = number of firms producing domestically ($M_d = M$ in closed economy). The integral in the numerator is:

$$
\int_{\phi_d^*}^{\infty} \phi^{\sigma-1} \, dG(\phi) = \int_{\phi_d^*}^{\infty} \phi^{\sigma-1} \cdot k \phi_{min}^k \phi^{-(k+1)} d\phi\\
= k \phi_{min}^k \int_{\phi_d^*}^{\infty} \phi^{\sigma - 1 - k - 1} \, d\phi = k \phi_{min}^k \int_{\phi_d^*}^{\infty} \phi^{-(k - \sigma +1)} \, d\phi
$$
Solving the integral:
$$
= k \phi_{min}^k \left[ \frac{\phi^{-(k-\sigma)}}{-(k-\sigma)} \right]_{\phi_d^*}^{\infty} = \frac{k}{k - \sigma +1} \phi_{min}^k (\phi_d^*)^{-(k - \sigma)}
$$


Now, the probability of surviving (being active):
$$
1 - G(\phi_d^*) = \left( \frac{\phi_{min}}{\phi_d^*} \right)^k
$$

---

<style scoped>
section {
    font-size: 27px;
}
</style>

So, the **average** $\phi^{\sigma-1}$ among active firms is:
$$
\frac{1}{1-G(\phi_d^*)} \int_{\phi_d^*}^{\infty} \phi^{\sigma-1} g(\phi) d\phi = \frac{k}{k-\sigma+1} \left( \frac{\phi_{min}^k}{(\phi_d^*)^k} \right) (\phi_d^*)^{\sigma-1} \cdot (\phi_d^*)^{k-\sigma}
$$


Simplifying:
$$
= \frac{k}{k - \sigma +1} \, (\phi_d^*)^{\sigma-1}
$$


Therefore, the aggregate productivity becomes:
$$
\tilde{\phi} = \left[ \frac{k}{k - \sigma +1} \right]^{\frac{1}{\sigma-1}} \phi_d^*
$$

---

**Final Clean Formula**:

<style scoped>
section {
    font-size: 27px;
}
</style>

$$
\tilde{\phi} = \left( \frac{k}{k - \sigma +1} \right)^{\frac{1}{\sigma-1}} \phi_d^*
$$


This shows that aggregate productivity $\tilde{\phi}$ is **proportional to the cutoff** $\phi_d^*$.

------

#### Important Implications

<style scoped>
section {
    font-size: 27px;
}
</style>

- Since $\tilde{\phi} \propto \phi_d^*$, anything that **raises the domestic cutoff** $\phi_d^*$ increases average industry productivity.
- When trade opens:
  - $\phi_d^*$ **rises** (least productive firms exit)
  - $\tilde{\phi}$ **increases** → this is the **reallocation gain**

This productivity gain is **new** compared to the Krugman model and can be quantitatively large.