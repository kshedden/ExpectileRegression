using ExpectileRegression
using LinearAlgebra
using Statistics
using UnicodePlots

n = 100
p = 4

X = randn(n, p)
b = randn(p)
lp = X * b
y = lp + rand(n)

# Compute pseudo-observations at beta'*x0
xt = randn(p)

m0 = fit(ExpectReg, X, y; tau=[0.5])

b0 = coef(m0)[:]
yt = dot(b0, xt)

X0 = copy(X[1:end-1, :])
y0 = copy(y[1:end-1])

H = zeros(p, p)
ExpectileRegression.expectreg_loss_hess!(m0, H, b0)
H ./= n

grad_obs = zeros(p, n)
ExpectileRegression.expectreg_loss_grad_obs!(m0, grad_obs, b0)

bf = zeros(n)
aa = zeros(n)
for i in 1:n

    # Brute-force
    ii = vcat(1:(i-1), (i+1):n)
    X0 .= X[ii, :]
    y0 = y[ii]
    m1 = fit(ExpectReg, X0, y0; tau=[0.5])
    b1 = coef(m1)
    bf[i] = n*yt - (n-1)*dot(b1, xt)

    # Linear approximation
    aa[i] = dot(xt, b0 - H\grad_obs[:, i])
end

xx = ones(n, 2)
xx[:, 2] = aa
ml = lm(xx, bf)
println(ml)

plt = scatterplot(aa, bf)
println(plt)
