# Risk aware controller
Portfolio optimization always involves certain risks. It is a well-known fact that targeting a specific level of risk is an important goal for any hedge fund. However, because of the need for constant rebalancing of the portfolio due to market movements, achieving this target can be quite costly. The aim of this research is to analyze a possible strategy for controlling risk, primarily in terms of Value at Risk (VaR). Is it possible to control the portfolio in such a way that the level of risk remains constant?

To achieve this goal, we consider a portfolio consisting of two assets: cash and a risky asset. We use a AR(1) - GARCH(1,1) model to calculate the VaR of the portfolio. Then, the Mean-Portfolio Contribution (MPC) is employed to target a specific VaR level while minimizing transaction costs.

The following Loss is optimized:
$$L = \alpha*(action*VaR_{t+1} - target VaR)^2 +(action - volume)^2$$

where 
1) $VaR_{t+1} = r_{t+1} + t.ppf(0.05, \nu)*\sigma_{t+1}$
1) $Volume  = \frac{prev \space action*(1+r_{t})}{((1-prev \space action)+(prev \space action-prev \space action*(1+r_{t}))}$

1) $\alpha$ is the weight of risk targeting

1) $r_{t+1},\sigma_{t+1}, \nu$ are specified by AR(1) - GARCH(1,1) by maximum likelyhood method 
In more details about VaR, MPC and backtesting, it can be read in `Theory_behind.pdf`


I compare the above method with two border baselines - first baseline is portfolio wighout any adjustment to risk - it will not suffer from transaction costs, but by construction it would fail to target risk. 

Second border baseline is risk parity portfolio, when transaction costs are not taken into account, in this case risk would be perfectly targeted, but PnL will greatly suffer from transaction costs.

In `main.ipynb` it can be seen the comparison

