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

### Global Value Chains

#### 全球價值鏈

**Wen-Bin Chuang**
**2026-09-14**

![bg right fit](./images/02201.jpg)

## 

----

<style scoped>
section {
    font-size: 27px;
}
</style>

`Global Value Chains (GVCs)` refer to the **fragmentation of production processes** across different countries. Instead of producing a good entirely in one country, companies break down the production into stages (design, raw materials, components, assembly, marketing, etc.) and locate each stage in the most cost-efficient country.

**Example**: An iPhone is designed in the USA, contains chips from Taiwan/Korea, rare earths from Australia/China, is assembled in China/Vietnam, and sold worldwide.

---

#### **Key Features**

<style scoped>
section {
    font-size: 27px;
}
</style>

- **Trade in Tasks** rather than finished goods
- Dominated by **Multinational Corporations (MNCs)** and their suppliers
- High share of **intermediate goods** in total trade (**≈ 60-70%** of world trade)
- Strong emphasis on **trade in services** (logistics, engineering, R&D, finance)

----

#### **Economic Impacts**

<style scoped>
section {
    font-size: 27px;
}
</style>

**Positive**:

- Efficiency Gains
  - `Specialization` by stage → `higher productivity`
  - Lower production costs → lower consumer prices globally
- Terms-of-Trade and Income Effects
  - Developing countries gain by joining GVCs even without having absolute advantage in final goods.
- Productivity Spillovers
  - Foreign Direct Investment (FDI) brings `technology`, `management practices`, and `skills`.

-----

**Challenges**:

<style scoped>
section {
    font-size: 27px;
}
</style>

- Heavy dependence on foreign markets and suppliers (vulnerability to shocks)
  - This `deep interconnectedness` has also created significant vulnerability to external shocks such as gopolitical and trade shocks, environmental and natural disasters and economic and financial shocks.

- `Smile Curve`
  - High value captured at design (US/EU) and marketing ends; lower value in assembly (developing countries)

- Environmental costs due to `long-distance transportation`
- Gains skewed toward `capital owners` and `skilled labor` → rising `inequality` within countries.
- Risk of "race to the bottom" in labor and environmental standards

----

#### **Core Economic Theory**

<style scoped>
section {
    font-size: 27px;
}
</style>

GVCs are explained by a combination of:

- **Ricardian Comparative Advantage** (extended to tasks/stages)
- **New Trade Theory** (Krugman) – economies of scale + imperfect competition
- **Fragmentation Theory** (Jones & Kierzkowski, 1990)
- **Trade in Tasks** (Grossman & Rossi-Hansberg, 2008)

**Key Idea**: Production is unbundled into **tasks**, and each task is performed in the country with the lowest opportunity cost for that specific task.

----

### Fragmentation Theory and Trade in Tasks

<style scoped>
section {
    font-size: 25px;
}
</style>

The **Fragmentation Theory** and the concept of **Trade in Tasks** represent a major paradigm shift in `international economics`. They move beyond traditional trade models—which focus on `countries` exchanging `final, finished goods` (like wine for cloth)—to explain how modern `production` is `sliced up` across borders.

#### Fragmentation Theory

`Fragmentation` refers to the process of breaking down a production process into distinct `stages or tasks,` and locating those stages in different countries to take advantage of specific local comparative advantages (e.g., cheaper labor, specialized skills, or proximity to resources).

- **Example:** Instead of the U.S. or Germany producing an entire car domestically, the design is done in Germany, the engine is manufactured in Japan, the electronics are coded in India, and the final assembly takes place in Mexico or China.

-----

#### Trade in Tasks
<style scoped>
section {
    font-size: 25px;
}
</style>

Coined prominently by economists **Gene Grossman and Esteban Rossi-Hansberg** in their seminal 2008 paper *"Trading Tasks: A Simple Theory of Offshoring,"* this concept argues that what countries are really trading today are not just goods, but the specific *tasks* required to make those goods.

- A country might be a net exporter of "R&D tasks" and "high-level management tasks," while being a net importer of "assembly tasks" and "routine data-processing tasks."
- **The "Second Unbundling":** Economist Richard Baldwin describes this as the `Second Unbundling.` The *First Unbundling* (Industrial Revolution) separated production from consumption (goods could be made far away from where they were consumed). The *Second Unbundling* (Information Age) separates production stages from one another.

----

##### Offshore makes a zero-sum game for the low-skills workers in Home?

<style scoped>
section {
    font-size: 25px;
}
</style>

- Before Grossman & Rossi-Hansberg (2008), the standard trade theory (the Stolper-Samuelson theorem) dictated a grim conclusion: If a rich country offshores low-skill tasks to a poor country, the rich country's low-skill workers will absolutely lose. It was viewed as a `zero-sum game`.

- GRH wanted to challenge this dogma. They wanted to prove that offshoring *could* actually **raise** the wages of the very low-skill workers whose jobs were being offshored. 
  - Through the **Productivity Effect** hidden inside the $\hat{P}_M$ equation. They mathematically demonstrated that offshoring acts exactly like a **positive technological shock**. It lowers the cost of production so drastically that the firm `expands its scale`, increasing the demand for the *remaining* domestic low-skill workers.

- In such case, offshoring can be a `Pareto improvement` where both high-skill and low-skill workers gain—relies entirely on the `Productivity Effect` overwhelming the `Labor Supply Effect` in the price equation.

----

<style scoped>
section {
    font-size: 27px;
}
</style>

- Imagine you only look at `Partial Equilibrium (just one firm)`. Firm A offshores tasks, fires 100 workers. The workers are poorer. The story ends. 

- But in the real world (General Equilibrium), the story doesn't end there.

1. Firm A `offshores`, lowering its costs.
2. Because costs are lower, Firm A lowers the price of its final good ($P_M$ falls).
3. Because the good is cheaper, global consumers `buy more` of it.
4. To meet this massive new global demand, Firm A has to `scale up production`.
5. Firm A `hires new domestic workers` for the tasks it kept at home.

----

##### Grossman & Rossi-Hansberg theory of Trade in Tasks(2008)

###### The Setup: Tasks, Costs, and Offshoring Frictions

<style scoped>
section {
    font-size: 25px;
}
</style>

- Consider a Home country producing a manufactured good $M$. Production of $M$ requires `a continuum of` tasks indexed by $z\in[0,1]$. 
  - Let $a_L(z)$ and $a_H(z)$ be the `unit labor requirements` for low-skill ($L$) and high-skill ($H$) labor to perform task $z$.
  - The `domestic (Home)` unit cost of performing task $z$ is: $c(z)=w_Ha_H(z)+w_La_L(z)$
  - The `foreign (Foreign)` unit cost of performing task $z$ is: $c^∗(z)=w_H^∗a_H(z)+w_L^∗a_L(z)$ 

- **The Offshoring Friction:** `Offshoring` is **not free**. 
  - Let $\tau(z)≥1$  represent the `iceberg offshoring cost` for task $z$ (where $\tau(z)=1$ means frictionless offshoring, and $\tau(z)>1$ means communication/coordination costs). 
  - The `cost to Home` of offshoring task $z$ to Foreign is $\tau(z)c^∗(z)$ .


-----

#### The Offshoring Margin (The "Chop Line")

<style scoped>
section {
    font-size: 25px;
}
</style>

- A profit-maximizing firm will offshore task $z$ if it is cheaper to do so abroad: $\tau(z)c^∗(z)<c(z)$ 

- Define the `relative cost advantage` of Foreign for task $z$ as $\Omega(z)=\frac{c(z)}{\tau(z)c^∗(z)}$ . 

  - Assume tasks are ordered such that $\Omega(z)$ is `strictly decreasing` in $z$. 
    - This means `low-index tasks` (e.g., $z\in[0,z^∗]$ ) are `offshored`, and high-index tasks (e.g., $z\in[z^∗,1]$ ]) are kept at Home.

- The **marginal task** $z^∗$  (the `chop line`) is defined by the `indifference condition`: 

  - $$
    c(z^∗)=\tau(z^∗)c^∗(z^∗)  ⟹  \Omega(z^∗)=1
    $$


----

#### The Unit Cost Function and the Envelope Theorem

<style scoped>
section {
    font-size: 25px;
}
</style>

- Because production requires `all` tasks, and tasks are combined in a `Cobb-Douglas continuum`, the `unit cost (price of the final good)` of good $M$ in `Home` is: 

  - $$
    P_M=exp⁡\left(\int_0^{z^∗}ln⁡[\tau(z)c^∗(z)]dz+\int_{z^∗} ^1ln⁡c(z)dz\right)
    $$

- Taking the `natural log`:

  - $$
    ln⁡P_M=\int_0^{z^∗}ln⁡[\tau(z)c^∗(z)]dz+\int_{z^∗} ^1ln⁡c(z)dz)
    $$

- **Discrete Cobb-Douglas**: $Y=\sum_{i=1}^N x_i^{\alpha_i}$,
- **Continuous Cobb-Douglas**: $Y=exp\left(\int_0^1 lnx(z)dz\right)$


----

#### The Crucial Mathematical Insight (The Envelope Theorem)

<style scoped>
section {
    font-size: 25px;
}
</style>

- What happens to the price $P_M$ if the offshoring margin $z^∗$ expands marginally (i.e., the firm offshores a tiny bit more)? 
  - Using `Leibniz's rule` to differentiate with respect to $z^∗$ : $\frac{\partial ln⁡P_M}{\partial z^∗}=ln⁡[\tau(z^∗)c^∗(z^∗)]−ln⁡c(z^∗)$ 
    - Leibniz's rule: 微上界, 代上界 - 微下界, 代下界
  - Because the marginal task $z^∗$ is chosen `optimally`, $\tau(z^∗)c^∗(z^∗)=c(z^∗)$ . Therefore: $\frac{\partial ln⁡P_M}{\partial z^∗}=0$

- **Economic Meaning:** A marginal endogenous shift in the offshoring margin has **zero first-order effect** on unit costs. The firm is already optimizing. Therefore, any change in $P_M$ must come from `exogenous shocks` (like a drop in $\tau$ or foreign wages).

----

#### Deriving the Three Effects

<style scoped>
section {
    font-size: 25px;
}
</style>

Let an `exogenous shock` occur: 

- offshoring costs $\tau(z)$ fall for the tasks currently being offshored. We use the hat notation ($\hat{x}=dx/x$) for percentage changes.

- `Total differentiating` $ln⁡P_M$  (and noting the $dz^∗$ terms cancel out): $\hat{P}_M=\int_0^{z^∗}\hat{\tau}(z)s(z)dz$  where $s(z)$ is the cost share of task $z$. ($d(lnx)=\frac{dx}{x}=\hat{x}$)

  - Differentiating the first integral: $\int_0^{z^*}s(z)[\hat{\tau}(z)+\hat{c}^{*}(z)]dz+s(z^*)ln[\tau(z^*)c^*{z^*}]dz^*$, 
  - Differentiating the second integral: $\int_{z^*}^1s(z)\hat{c}(z)dz-s(z^*)ln[c(z^*)]dz^*$ 

  $$
  \hat{P}_M=\int_0^{z^*}s(z)[\hat{\tau}(z)+\hat{c}^{*}(z)]dz+\int_{z^*}^1s(z)\hat{c}(z)dz
  $$

----

<style scoped>
  section {
      font-size: 25px;
  }
</style>


- We can do this by adding and subtracting the term: $\int_0^{z^*}s(z)\hat{c}(z)dz$

  - $$
    \hat{P}_M=\underbrace{\int_0^{z^*}s(z)\hat{\tau}(z)dz}_{term1} +\underbrace{\int_0^{z^*}s(z)[\hat{c}^{*}(z)-\hat{c}(z)]dz}_{term2}+\underbrace{\int_0^1s(z)\hat{c}(z)dz}_{term3}
    $$

- Now, let's look at the `domestic factor market clearing conditions` to isolate the three effects.

-----

###### A. The Productivity Effect

<style scoped>
section {
    font-size: 25px;
}
</style>

- The first term is the direct cost reduction from `cheaper offshoring`. 
  - We can hold `domestic` and `foreign` unit costs constant ($\hat{c}(z)=0, \hat{c}^*(z)=0$).  $\hat{P}_M=\int_0^{z^*}s(z)[\hat{\tau}(z)]dz$

- Because offshoring costs are falling, $\hat{\tau}(z)<0$ , Grossman and Rossi-Hansberg define the **Productivity Shock** ($\pi$) as the absolute value of this cost reduction: $\pi≡−\int_0^{z^∗}\hat{\tau}(z)s(z)dz>0$.
  - The change in the price of the `final good` is simply, term 1 is simply: $\hat{P}_M=−\pi$.


- **Economic Meaning:** This offshore acts exactly like **Hicks-neutral technological progress**. It is a pure, unadulterated reduction in the cost of production, independent of any wage changes.

------

##### The Domestic Zero-Profit Condition

<style scoped>
section {
    font-size: 25px;
}
</style>


- The third term represents the change in `domestic` unit costs across *all* tasks. 
- Recall that the unit cost of a task is a `weighted average` of factor prices: $\hat{c}(z)=\theta_L (z)\hat{w}_L+\theta_H (z)\hat{w}_H$. where $\theta_i(z)$  is the cost share of factor $i$ in task z.

- $\int_0^1s(z)\hat{c}(z)dz=\left(\int_0^1s(z)\theta_L (z)dz\right)\hat{w}_L+\left(\int_0^1s(z)\theta_H (z)dz\right)\hat{w}_H$
  - The integrals in the parentheses are simply the **aggregate cost shares** of `low-skill` and `high-skill labor` in the entire manufacturing sector. 
  - Let's call them $\theta_{LM}$  and $\theta_{HM}$, Term 3 simplifies to $\theta_{LM}\hat{w}_L+\theta_{HM}\hat{w}_H$.
- **Economic Meaning:** This is the standard `zero-profit condition component`. It shows how changes in domestic wages feed into the final price of the good.

----

##### B. The Relative Price Effect (Terms of Trade)

<style scoped>
section {
    font-size: 25px;
}
</style>


- The second term captures the difference in cost changes between `foreign` and `domestic tasks`. 
- If offshoring increases the demand for `foreign` low-skill labor, foreign wages ($\hat{w}_L^*$) will rise. 
  - Because foreign tasks are low-skill intensive, the foreign unit cost $\hat{c}^∗(z)$  will `rise faster` than the domestic unit cost $\hat{c}(z)$ .
  - Therefore, $[\hat{c}^{*}(z)-\hat{c}(z)]>0$, making Term 2 positive.
- **Economic Meaning:** This is the **Relative Price Effect** (or terms-of-trade effect). As foreign tasks become more expensive, the Home country experiences a slight deterioration in its terms of trade, which offsets some of the productivity gains and puts downward pressure on domestic factor returns.

-----

##### Synthesizing the Price Equation

<style scoped>
section {
    font-size: 25px;
}
</style>


- Substituting our definitions back into the decomposed equation, we get the **General Equilibrium Zero-Profit Condition** for sector $M$:
  $$
  \hat{P}_M=-\pi+\underbrace{\int_0^{z^*}s(z)[\hat{c}^{*}(z)-\hat{c}(z)]dz}_{\text{Relative Price Effect(+)}}+ \theta_{LM}\hat{w}_L+\theta_{HM}\hat{w}_H
  $$

- Rearranging to solve for the `domestic factor payments`:
  $$
  \theta_{LM}\hat{w}_L+\theta_{HM}\hat{w}_H=\hat{P}_M+\pi-\text{Relative Price Effect}
  $$

  - This equation tells us that the total income paid to domestic factors in sector $M$ is driven by 
    - `the price of the good` ($\hat{P}_M$), 
    - boosted by the `Productivity Effect` ($\pi$), 
    - reduced by the `Relative Price Effect.

-----

##### The Left-Hand Side: The "Domestic Wage Pie"

<style scoped>
section {
    font-size: 25px;
}
</style>


- **Economic Meaning:** This is the **percentage change in the total value-added per unit of output** that is available to be distributed to domestic workers. 
 - Because $\theta$ represents the cost share of each type of labor, this weighted average tells us how much the overall "pie" of domestic factor income is growing (or shrinking). 
 - If this number is positive, domestic workers, as a collective group in this sector, are getting richer.



###### $\hat{P}_M$  : The Baseline Revenue Effect

- **Economic Meaning:** This is the **change in the price of the final good**. In a standard, closed-economy trade model (with no offshoring), this is the *only* term that matters. 
  - If the price of the good you sell goes up by 5%, your revenue goes up by 5%, and you can afford to pay your workers 5% more. It is the baseline, traditional source of wage growth.

-----

##### $+\pi$: The "Offshoring Dividend" (The Productivity Effect)

<style scoped>
section {
    font-size: 25px;
}
</style>

- **Economic Meaning:** This is the **cost savings generated by accessing cheaper global inputs**. 
  - When a firm offshores tasks, it is essentially importing foreign services. If the cost of coordinating with those foreign tasks falls (e.g., cheaper internet, better logistics, lower tariffs represented by a drop in $\tau$), the firm's overall cost of production drops. 

- **The Intuition:** This cost reduction acts exactly like a **technological breakthrough**. 
  - The firm is now more efficient. Even if the final price of the good ($\hat{P}_M$) stays exactly the same, the firm is spending `less` to produce it. 
  - That extra margin doesn't just vanish into thin air; in a competitive market, it is passed on to the domestic factors of production. Therefore, `cheaper offshoring` directly **boosts** the domestic wage pie.

-------

###### $-\text{Relative Price Effect}$: The "Foreign Catch-Up" Penalty

<style scoped>
section {
    font-size: 25px;
}
</style>

- **Economic Meaning:** This is the **erosion of offshoring savings due to rising foreign costs** (often tied to the Terms of Trade). 
  - When the Home country offshores a massive amount of tasks, it creates a huge derived `demand` for foreign labor. This increased demand drives up foreign wages ($\hat{w}^∗$), which in turn drives up the cost of producing those tasks abroad ($\hat{c}^∗$).

- **The Intuition:** Imagine you start buying all your coffee beans from a new, cheap supplier. 
  - At first, your costs plummet (the Productivity Effect). But because you are buying `so much` coffee from them, the supplier realizes they have pricing power and raises their prices. Your savings are partially eaten away.

----

###### The Big Picture: A Real-World Analogy (Apple and the iPhone)

<style scoped>
section {
    font-size: 25px;
}
</style>


Imagine Apple (the Home country) designing the iPhone.

1. **Baseline Revenue** ($\hat{P}_M$): If Apple can sell the iPhone for a `higher price`, it has more money to pay its US engineers and managers.
2. **The Offshoring Dividend** ($+\pi$): Apple figures out how to seamlessly coordinate with Foxconn in China (lower $\tau$). Because this coordination is so cheap and efficient, Apple's overall cost to build the phone drops drastically. This massive efficiency gain allows Apple to pay its US designers and software engineers *more*, even if the price of the iPhone stays the same. **This is the magic of the Productivity Effect.**
3. **The Foreign Catch-Up Penalty** ($−\text{Relative Price Effect}$): However, because Apple and every other US tech company are offshoring so much to China, the demand for Chinese factory workers skyrockets. Chinese wages begin to rise rapidly. Foxconn has to charge Apple more to assemble the phones. This rising foreign cost eats into some of the savings Apple initially enjoyed, slightly dampening the wage growth of the US engineers.

----

#### Summary

<style scoped>
section {
    font-size: 25px;
}
</style>


The equation mathematically proves that **offshoring is not just about moving jobs**. It is about **redefining the domestic production function**. By slicing up the value chain, a country transforms its domestic wage growth from being dependent solely on the price of its final goods ($\hat{P}_M$), to being driven by its ability to efficiently orchestrate global networks ($+\pi$), net of the cost of bidding up global resource prices ($−\text{Relative Price Effect}$).



By breaking the change in $P_M$ into the **Three Effects**, economists can isolate these competing forces:

- **The Labor Supply Effect** says: "Offshoring destroys domestic jobs, pushing wages **down**."
- **The Productivity Effect** says: "Offshoring makes the firm hyper-efficient, pushing wages **up**."
- **The Relative Price Effect** says: "Offshoring makes us rely on foreigners, whose wages are now rising, pushing our terms of trade and wages **down**."

----

#### C. The Labor Supply Effect

<style scoped>
section {
    font-size: 25px;
}
</style>

- It operates through the **factor market clearing conditions**. 
- Let's look at the `domestic demand` for `low-skill labor` ($D_L^M$). As derived previously, when the offshoring margin $z^∗$ `expands`, domestic labor demand falls:
$$
\frac{\partial D_L^M}{\partial z^*}=-x_Ma_L(z^*)<0
$$
- GRH use the **Effective Endowment Trick**. They define the `effective domestic endowment` of `low-skill labor` as:
$$
\bar{L}_{eff}=\bar{L}+x_M \int_0^{z^*} a_L^*(z)\tau(z)dz
$$
- When $z^∗$  increases, the integral grows, meaning $\bar{L}_{eff}$ increases. By the **Rybczynski Theorem**, an increase in the effective endowment of $L$ relative to $H$ will `lower` the relative price of $L$.

-----

<style scoped>
section {
    font-size: 25px;
}
</style>

- To find the final impact on the real wage of low-skill labor ($\hat{w}_L$), we combine the zero-profit condition (which gives us the Productivity and Relative Price effects) with the factor market clearing condition (which gives us the Labor Supply effect). 
- Grossman and Rossi-Hansberg arrive at the final, famous decomposition:
$$
\hat{w}_L=\underbrace{\frac{\pi}{\theta_{LM}}}_{Productivity Effect}-\underbrace{\text{Labor Supply Effect}}_{DirectDisplacement}-\underbrace{\text{Relative Price Effect}}_{Terms of Trade}
$$

----
<style scoped>
section {
    font-size: 25px;
}
</style>
- For the `low-skill worker` to actually `benefit`, we require $\hat{w}_L>0$. Therefore, the condition is: $\frac{\pi}{\theta_{LM}}> (\text{Labor Supply Effect}+\text{Relative Price Effect})$ 
  - The **Productivity Effect** emerges directly from the $\hat{\tau}(z)$ integral.
  - The **Relative Price Effect** emerges from the divergence between foreign and domestic cost changes ($\hat{c}^∗−\hat{c}$ ).
  - The **Labor Supply Effect** is isolated by mapping the drop in domestic labor demand to an increase in the effective factor endowment. 

---
<style scoped>
section {
    font-size: 25px;
}
</style>  
- **What this means:** The cost savings generated by cheaper offshoring ($\pi$) must be `large enough` to completely `absorb` the `wage-depressing effects` of direct job displacement and rising foreign costs.
- In fact, When $z^∗$ is `small`, the right side of the inequality (the negative effects) approaches zero, while the left side ($\pi$) remains positive. Therefore, $\hat{w}_L>0$. 
  - The low-skill workers whose tasks are *just barely* being offshored actually see their wages **rise** because the massive efficiency gains of the firm expand the overall wage pie, and there is almost no displacement penalty to pay.

----

Why Analyzing $P_M$ is the Core of the GRH (2008) Breakthrough:

<style scoped>
section {
    font-size: 25px;
}
</style>

- **Challenging the Old Dogma:** Pre-2008 trade theory (Stolper-Samuelson) assumed offshoring was a `zero-sum game` that `inevitably` hurt low-skill workers in rich countries.
- **The "Win-Win" Mechanism:** GRH used the $P_M$ (price) equation to prove this wrong. By isolating the **Productivity Effect** ($\pi$), they mathematically showed that offshoring acts as a **positive technological shock**.
- **The General Equilibrium Feedback Loop:** This productivity shock triggers a macroeconomic chain reaction: lower production costs →→ lower final good prices ($P_M$ falls) →→ higher global consumer demand →→ firms scale up production →→ **increased demand and wages for the remaining domestic low-skill workers.**
- **The Bottom Line:** If you don't track $P_M$ and isolate the Productivity Effect, you miss this expansionary feedback loop. You cannot mathematically prove the paper's groundbreaking conclusion: that offshoring can be a `Pareto improvement` that actually `benefits` the very workers whose tasks are being moved abroad.
