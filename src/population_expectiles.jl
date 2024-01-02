# Returns the value tau\in (0,1) such that e is the
# tau'th expectile of a standard normal distribution.
function normal_invexpectile(e)
    a = e*cdf(Normal(), e) + pdf(Normal(), e)
    return a / (2*a - e)
end

# Returns the tau'th expectile of a standard normal distribution.
function normal_expectile(tau)
    return find_zero(x->normal_invexpectile(x)-tau, (-10, 10))
end

# Returns the value tau\in (0,1) such that e is the
# tau'th expectile of a standard student T distribution
# with df degrees of freedom.
function t_invexpectile(e, df)
    u = (df + e^2) * pdf(TDist(df), e) / (1 - df) - e * cdf(TDist(df), e)
    return u / (2 * u + e)
end

# Returns the tau'th expectile of a standard normal distribution.
function t_expectile(tau, df)
    return find_zero(x->t_invexpectile(x, df)-tau, (-20, 20))
end
