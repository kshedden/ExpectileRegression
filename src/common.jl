# The check function
function check(y, tau)
    return y > 0 ? tau : 1 - tau
end

# The expectile loss function.
function eloss(x, tau)
    f = x > 0 ? tau : 1 - tau
    return f * x * x
end

# The derivative of the expectile loss function (score function).
function elossgrad(x, tau)
    f = x > 0 ? tau : 1 - tau
    return 2 * f * x
end
