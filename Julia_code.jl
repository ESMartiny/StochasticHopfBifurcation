using LinearAlgebra
using Plots
using DifferentialEquations
using ParameterizedFunctions
using FFTW
using LsqFit
using StatsBase

gluco = @ode_def begin
  dx = -x + a*y + x^2*y
  dy = b - a*y - x^2*y
end a b σ

chlorine = @ode_def begin
    dx = a - x - 4*x*y/(1+x^2)
    dy = b*x*(1-y/(1+x^2))
end a b

brusselator = @ode_def begin
    dx = a + y*x^2 - b*x - x
    dy = b * x - y * x^2
end a b

normal_form = @ode_def begin
    dx = x*μ - ω*y - (a*x + y*b)*(x^2+y^2)
    dy = x*ω + y*μ - (a*y - b*x)*(x^2+y^2)
end a b μ ω σ

noise_term_2d = @ode_def begin
    dx = σ
    dy = σ
end a b σ

noise_term_2d_4_var = @ode_def begin
    dx = σ
    dy = σ
end a b μ ω σ

x_0 = [.5,1.4]
t_span = (0.0,2000.0)
p = (0.113,0.481,0.01)
prob_sde_gluco = SDEProblem(gluco, noise_term_2d, x_0,t_span,p)
sol = solve(prob_sde_gluco, ISSEM())
histogram2d(sol,vars=(1,2),bins=(0.4:0.005:0.6,1.3:0.005:1.5))
xlims!(0.4,0.6)
histogram(time_emb_r(vec(sol[1,:])),bins=100)
plot(sol)

rs = time_emb_r(vec(sol[1,:]),p[2])
rs_sorted = sort(rs)
ndata = 2.5
n = floor(Int64,length(rs_sorted)/ndata)
histogram(rs_sorted[1:n],bins=60)
f = fit_second_order(sol,p[2],2.5, 60)


function sim_plus_fit(x_0, t_span, fun, f_σ, p, ndata, nbins)
    prob_sde_gluco = SDEProblem(fun, f_σ, x_0, t_span, p)
    sol = solve(prob_sde_gluco, SOSRA2())
    f = fit_second_order(sol,p[2],ndata,nbins)
    return (f.param[2], confidence_interval(f, 0.05)[2][1],confidence_interval(f, 0.05)[2][2])
end

function sim_plus_kolmo(x_0, t_span, fun, f_σ, p, ndata, nbins)
    prob_sde_gluco = SDEProblem(fun, f_σ, x_0, t_span, p)
    sol = solve(prob_sde_gluco)
    f = kol_smirnov(sol,p[2],ndata,nbins)
    return f
end

function raster_scan(x_0, t_span, fun, f_σ, b_range, σ_range, ndata, nbins)
    data = zeros(length(b_range),length(σ_range))
    data_low = zeros(length(b_range),length(σ_range))
    data_up = zeros(length(b_range),length(σ_range))
    for (i, b) in enumerate(b_range)
        for (j, σ) in enumerate(σ_range)
            p = (0.1, b, σ)
            f = sim_plus_fit(x_0, t_span, fun, f_σ, p, ndata, nbins)
            data[i,j] = f[1]
            data_low[i,j] = f[2]
            data_up[i,j] = f[3]
        end
        show(b)
    end
    return data,data_low,data_up
end

b_range = (0.4:0.002:0.44)
σ_range = (0.001:0.0010:0.010)
x_0 = [1.0,1.0]
t_span = (0.0,100000.0)
@time data,data_low,data_up = raster_scan(x_0, t_span, gluco, noise_term_2d, b_range, σ_range, 3, 60)

levels=[-5,-2,-1,0,1,2,5,10]
contourf(σ_range, b_range,data,levels=[0,.05,0.1,0.15,0.2,0.5])
contourf(σ_range, b_range, data)
xticks!(σ_range)
yaxis!(b_range)
plot(0.38:0.005:0.45,lower_list[31:end])
plot!(0.38:0.005:0.45,upper_list[31:end])
plot!(0.38:0.005:0.45,ks_stat[31:end])
ks_stat

fi = fit_second_order_1var(sol,p[2],10,50)
 p[1]*x^2 + ($max_val/$l - p[1]*$l)*x)


function kol_smirnov(sol,f,ndata, nbins)
    rs = time_emb_r(vec(sol[1,:]),f)
    rs_sorted = sort(rs)
    n = floor(Int64,length(rs_sorted)/ndata)
    h = fit(Histogram, rs_sorted[1:n],nbins=nbins)
    l = length(h.weights)-1
    empi = h.weights[1:end-1]
    max_val = empi[end]
    linear = [n * (max_val/l) for n in 1:l]
    empi = empi ./ sum(empi)
    linear = linear ./ sum(linear)
    ks_stat = 0
    for n in range(1, l)
        ks_stat = max(ks_stat, abs(sum(empi[1:n]) - sum(linear[1:n])))
    end
    return ks_stat,empi,linear
end
function fit_second_order(sol, f, ndata, nbins)
    rs = time_emb_r(vec(sol[1,:]),f)
    rs_sorted = sort(rs)
    n = floor(Int64,length(rs_sorted)/ndata)
    h = fit(Histogram, rs_sorted[1:n],nbins=nbins)
    empi = h.weights[1:end-1]
    @. model(x, p) = p[1] * x + p[2]*x^2+1
    a = (empi[end]/(length(empi)-1))
    xdata = 0:length(empi)-1
    p0 = [a,0]
    lb = [0, -Inf]
    ub = [length(empi), Inf]
    #f = curve_fit(model2, xdata, empi, p0,lower=lb, upper=ub)
    f = curve_fit(model, xdata, empi, p0)
    return f
end
function fit_second_order_1var(sol, f, ndata, nbins)
    rs = time_emb_r(vec(sol[1,:]),f)
    rs_sorted = sort(rs)
    n = floor(Int64,length(rs_sorted)/ndata)
    h = fit(Histogram, rs_sorted[1:n],nbins=nbins)
    empi = h.weights[1:end-2]
    max_val = empi[end]*1.0
    l = length(empi)-1
    ex = :(@. model(x, p) = p[1]*x^2 + ($max_val/$l - p[1]*$l)*x)
    eval(ex)
    xdata = 0:length(empi)-1
    p0 = [0.0]
    #f = curve_fit(model2, xdata, empi, p0,lower=lb, upper=ub)
    f = curve_fit(model, xdata, empi, p0)
    return f
end
function fourier(solution,t_span )
    N = length(solution[1,:])
    dt = (t_span[2]-t_span[1])/(N)
    F = fft(solution[1,:])

end
function time_emb_poincare(time_series)
    corr = StatsBase.autocor(time_series,range(1,stop=2000))
    τ = findfirst(corr.<0)
    x = time_series[1:end-τ]
    xτ = time_series[τ+1:end]
    side = (xτ .> x)
    crosses = side[1:(end-1)] .⊻ side[2:end]
    time_series = time_series[1:length(crosses)]
    return time_series[crosses]
end
function time_emb_r(time_series,f)
    n = 1
    corr = StatsBase.autocor(time_series, [n])
    while corr[1] > 0
        n += 1
        corr = StatsBase.autocor(time_series, [n])
    end
    #τ = findfirst(corr.<0)
    τ = n
    x = time_series[1:end-τ]
    xτ = time_series[τ+1:end]
    r(x1,x2) = sqrt((x1-f)^2 + (x2-f)^2)
    rvec = map(r,x,xτ)
end





F1, F2 = fourier(sol,t_span)
F1_4, F2 = fourier(sol,t_span)
plot([1000:10:10000],broadcast(abs,real(F1_3[1000:10:10000])),xaxis=:log,yaxis=:log)
plot!(broadcast(abs,real(F1[800:2000])),xaxis=:log,yaxis=:log)
plot!(real(F2[2:10000]))

F1= fftshift(F1)

(-a)/(b^4-a^2)
2b^2/(a+b^2)^2

μ_calc(a,b) = 2b^2/(a+b^2)-1
μ_calc2(a,b) = -b^2-a

μ_calc(0.1,0.31)
μ_calc(0.1,0.42)
a = 0.1
bs = 0.45
τ(a,b) = (b^2-a-2a*b^2-a^2-b^4)/(b^2+a)/2
τ(a,bs)
plot(0.3:0.001:0.52,broadcast(τ, 0.1, (0.3:0.001:0.52)))

zeros(3,2)

b(a) = sqrt(1.0/2-a-sqrt(1-8*a)/2)

b(0.113)
