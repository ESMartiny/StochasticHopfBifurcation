include("./Funcs.jl")
using .Funcs

using LinearAlgebra
using Plots
using DifferentialEquations
using ParameterizedFunctions
using FFTW
using LsqFit
using StatsBase
using LaTeXStrings
normal_form = @ode_def begin
    dx = x*μ - ω*y - (a*x + y*b)*(x^2+y^2)
    dy = x*ω + y*μ - (a*y - b*x)*(x^2+y^2)
end a b μ ω σ
noise_term_2d_4_var = @ode_def begin
    dx = σ
    dy = σ
end a b μ ω σ

#normal_form
x_0 = [.0,.0]
t_span = (0.0,10.0)
p=(1,1,-0.00,10,0.1)
ndata=3
nbins =60
prob_sde_gluco = SDEProblem(normal_form,noise_term_2d_4_var, x_0,t_span,p)
@time sol = solve(prob_sde_gluco, ISSEM())
plot(sol,vars=(1),)
xlabel!("Time")
ylabel!("X(t)")
histogram(sol[1,:],bins=100)
length(sol)

# brusselator
brusselator = @ode_def begin
    dx = a + y*x^2 - b*x - x
    dy = b * x - y * x^2
end a b σ

noise_term_2d = @ode_def begin
    dx = σ
    dy = σ
end a b σ

noise_term_2d_multi = @ode_def begin
    dx = σ * sqrt(x)
    dy = σ * sqrt(y)
end a b σ
x_0 = [1.0,1.0]
t_span = (0.0,100.0)
p=(1, 2.15, 0.05)
ndata=10
nbins =60
prob_sde_brusselator = SDEProblem(brusselator,noise_term_2d_multi, x_0,t_span,p)
@time sol = solve(prob_sde_brusselator, SROCK2();dt=0.01)
t_span_long = (0.0,20000.0)
prob_sde_brusselator_long = SDEProblem(brusselator,noise_term_2d_multi, x_0,t_span_long,p)
@time sol_long = solve(prob_sde_brusselator_long, SROCK2();dt=0.05)

plot(sol,vars=(1),color="red",linewidth=4, label="")

plot!(xguide=L"t",guidefontsize=34)
plot!(yguide=L"X",guidefontsize=30)
xticks!([0,50,100])
yticks!([1,1.5])
#plot!(annotations=[ (0,2.2, "a")])
plot!(legend=false, grid=false, legendfontsize=16,
                xtickfontsize=26, ytickfontsize=26, margin=6Plots.mm)
plot!(size=(750,500))
savefig("C:/Users/Martiny/Dropbox/PHD/Speciale_artikel/brusselator_timeserie_LCO.pdf")
x_l,xτ_l = Funcs.time_emb(vec(sol_long[1,:]),p[1])
x,xτ = Funcs.time_emb(vec(sol[1,:]),p[1])
crosses = Funcs.time_emb_poincare_filter(vec(sol_long[1,:]))
sort!(crosses)
plot(crosses)


histogram2d(x_l,xτ_l,normed=true)
plot!(x,xτ,linewidth=0.5,color="red",alpha=0.8,label="")
plot!(0.5:0.1:2,0.5:0.1:2,linewidth=6,label="")
scatter!([quantile(crosses,0.5)],[quantile(crosses,0.5)],markersize=12,color="white",label="")

plot!(xguide=L"X",guidefontsize=30)
plot!(yguide=L"Y",guidefontsize=30)
plot!(size=(750,500))
plot!(legend=:right, grid=false, legendfontsize=16,
                xtickfontsize=26, ytickfontsize=26,
                margin=6Plots.mm)
savefig("C:/Users/Martiny/Dropbox/PHD/Speciale_artikel/brusselator_timeemb_LCO.pdf")

#t_span_long = (0.0,10000.0)
#prob_sde_brusselator_long = SDEProblem(brusselator,noise_term_2d, x_0,t_span_long,p)
#@time sol = solve(prob_sde_brusselator_long, SROCK2();dt=0.05)
bins= 0.45:0.03:2.1
histogram(sol_long[1,:],bins=bins, normed=true, label="")
#xlabel!("X")
#ylabel!("P(X)")
function skewed_double_gauss(c, sig, fix,r,x)
    forst = c/(1+c)/sqrt(2*pi*sig^2)*exp(-(x-(fix-r))^2/(2*sig^2))
    anden = 1.0/(1+c)/sqrt(2*pi*sig^2)*exp(-(x-(fix+r))^2/(2*sig^2))
    return forst+anden
end
plot!(bins,[skewed_double_gauss(2.5,4*p[3],1,sqrt(0.075), x) for x in bins],linewidth=10,label="")
plot!(xguide=L"X",guidefontsize=30)
plot!(yguide=L"P(X)",guidefontsize=30)

xticks!([0.5,1.5])
yticks!([0.5,1.5])
plot!(legend=:right, grid=false, legendfontsize=16,
                xtickfontsize=26, ytickfontsize=26,
                margin=6Plots.mm)
plot!(size=(750,500))
savefig("C:/Users/martiny/Dropbox/PHD/Speciale_artikel/brusselator_xhisto_LCO.pdf")
rs = Funcs.time_emb_r(vec(sol_long[1,:]),p[1])
rs_sorted = sort(rs)
ndata=1
n = floor(Int64,length(rs_sorted)/ndata)
bins= 0:0.01:0.35
histogram(rs_sorted[1:n],bins=bins,normed=true, label="")
#xlabel!("R")
#ylabel!("P(R)")
plot!(bins, [b*27 for b in bins],color="orange",linewidth=8,label="")
h = fit(Histogram, rs_sorted[1:n],bins)
toppoint = argmax(h.weights)
l = floor(Int64,toppoint/ndata)
empi = h.weights[1:l]
@. model(x, p) = p[1] * x + p[2]*x^2
a = (empi[end]/(length(empi)-1))
xdata = 0:length(empi)-1
p0 = [a,0]
lb = [0, -Inf]
ub = [length(empi), Inf]
#f = curve_fit(model2, xdata, empi, p0,lower=lb, upper=ub)
f = curve_fit(model, xdata, empi, p0)
m = model(xdata, f.param)
m = m./m[end]*bins[end]*27
m = prepend!(m,0)
plot!(bins,m,linewidth=8,color="red",labels="")

plot!(xguide=L"R",guidefontsize=30)
plot!(yguide=L"P(R)",guidefontsize=30)
yticks!([2,6])
plot!(legend=:right, grid=false, legendfontsize=16,
                xtickfontsize=26, ytickfontsize=26,
                margin=6Plots.mm)
plot!(size=(750,500))
savefig("C:/Users/martiny/Dropbox/PHD/Speciale_artikel/brusselator_rhisto_LCO.pdf")
#gluco
gluco = @ode_def begin
  dx = -x + a*y + x^2*y
  dy = b - a*y - x^2*y
end a b σ

x_0 = [.4,1.6]
t_span = (0.0,150.0)
p = (0.1,0.4,0.01)
prob_sde_gluco = SDEProblem(gluco,noise_term_2d, x_0,t_span,p)
@time sol = solve(prob_sde_gluco, SROCK2();dt=0.01)
t_span_long = (0.0,15000.0)
prob_sde_gluco_long = SDEProblem(gluco,noise_term_2d, x_0,t_span_long,p)
@time sol_long = solve(prob_sde_gluco_long, SROCK2();dt=0.05)

plot(sol,vars=(1), color="red",linewidth=4, label="")
plot!(xguide=L"t",guidefontsize=34)
plot!(yguide=L"X",guidefontsize=30)
yticks!([0.4,0.5])
xticks!([0,75,150])

#plot!(title="Glycolyse SC", title_location=:left, titlefontcolor=:black, titlefontfamily=:Computer_Modern, titlefontsize=30 )
plot!(legend=:right, grid=false, legendfontsize=16,
                xtickfontsize=26, ytickfontsize=26,
                margin=6Plots.mm)
plot!(size=(750,500))
savefig("C:/Users/Martiny/Dropbox/PHD/Speciale_artikel/gluco_timeserie_NDO.pdf")
x,xτ = Funcs.time_emb(vec(sol[1,:]),p[1])
x_l,xτ_l = Funcs.time_emb(vec(sol_long[1,:]),p[1])
crosses = Funcs.time_emb_poincare(vec(sol_long[1,:]))
sort!(crosses)
plot(crosses)
histogram2d(x_l,xτ_l,normed=true)
plot!(x,xτ,linewidth=0.7,color="red",alpha=0.6,label="")
plot!(0.3:0.1:.5,0.3:0.1:0.5,linewidth=10,label="")
scatter!([quantile(crosses,0.5)],[quantile(crosses,0.5)],markersize=12,color="white",label="")
#scatter!([p[2]],[p[2]],markersize=8,label="")
#xlabel!("X(t)")
#ylabel!(L"X(t+\tau)")

plot!(xguide=L"X",guidefontsize=30)
plot!(yguide=L"Y",guidefontsize=30)
yticks!([0.3,0.4,0.5])
xticks!([0.3,0.4,0.5])
plot!(legend=:right, grid=false, legendfontsize=16,
                xtickfontsize=26, ytickfontsize=26,
                margin=6Plots.mm)
plot!(size=(750,500))
savefig("C:/Users/Martiny/Dropbox/PHD/Speciale_artikel/gluco_timeemb_NDO.pdf")
bins=0.25:0.005:0.55
histogram(sol_long[1,:],bins=bins, normed=true, label="")
#xlabel!("X")
#ylabel!("P(x)")
function single_gauss(sig, mean,x)
    return 1.0/sqrt(2*pi*sig^2)*exp(-(x-mean)^2/(2*sig^2))
end
plot!(bins,[single_gauss(4*p[3],p[2],b) for b in bins],linewidth=10, label="")

plot!(xguide=L"X",guidefontsize=30)
plot!(yguide=L"P(X)",guidefontsize=30)
xticks!([0.3,0.4,0.5])
yticks!([0,5,10])
plot!(legend=:right, grid=false, legendfontsize=16,
                xtickfontsize=26, ytickfontsize=26,
                margin=6Plots.mm)
plot!(size=(750,500))
savefig("C:/Users/Martiny/Dropbox/PHD/Speciale_artikel/gluco_xhisto_ndo.pdf")
bins=0.0:0.001:0.04
rs = Funcs.time_emb_r(vec(sol_long[1,:]),p[2])
rs_sorted = sort(rs)
ndata=1.1
n = floor(Int64,length(rs_sorted)/ndata)
histogram(rs_sorted[1:n],bins=bins, normed=true, label="")
#xlabel!("R")
#ylabel!("P(R)")
plot!(bins, [b*1075 for b in bins],color="orange",linewidth=8,label="")
h = fit(Histogram, rs_sorted[1:n],bins)
toppoint = argmax(h.weights)
l = floor(Int64,toppoint/ndata)
empi = h.weights[1:l+1]
@. model(x, p) = p[1] * x + p[2]*x^2
a = (empi[end]/(length(empi)-1))
xdata = 0:length(empi)-1
p0 = [a,0]
lb = [0, -Inf]
ub = [length(empi), Inf]
#f = curve_fit(model2, xdata, empi, p0,lower=lb, upper=ub)
f = curve_fit(model, xdata, empi, p0)
m = model(xdata, f.param)
m = m./m[end]*bins[end]*1000
#m = prepend!(m,0)
plot!(bins[1:length(m)],m,linewidth=8,color="red",labels="")

plot!(xguide=L"R",guidefontsize=30)
plot!(yguide=L"P(R)",guidefontsize=30)
yticks!([0,20,40])
xticks!([0,0.02,0.04])
plot!(legend=:right, grid=false, legendfontsize=16,
                xtickfontsize=26, ytickfontsize=26,
                margin=6Plots.mm)
plot!(size=(750,500))
savefig("C:/Users/Martiny/Dropbox/PHD/Speciale_artikel/gluco_rhisto_ndo.pdf")
histogram2d(sol,vars=(1,2))

# gluco melting peaks plot
x_0 = [.5,1.4]
t_span = (0.0,20000.0)
p_low = (0.1,0.436,0.005)
p_mid = (0.1,0.436,0.01)
p_high = (0.1,0.436,0.02)

gluco_low = SDEProblem(gluco,noise_term_2d, x_0,t_span_long,p_low)
gluco_mid = SDEProblem(gluco,noise_term_2d, x_0,t_span_long,p_mid)
gluco_high = SDEProblem(gluco,noise_term_2d, x_0,t_span_long,p_high)
@time sol_low = solve(gluco_low, SROCK2();dt=0.01)
@time sol_mid = solve(gluco_mid, SROCK2();dt=0.01)
@time sol_high = solve(gluco_high, SROCK2();dt=0.01)
bins = 0.2:0.01:0.7
histogram(sol_low[1,:],bins=bins, normed=true, label="")
histogram!(sol_mid[1,:],bins=bins, normed=true,alpha=0.7, label="")
histogram!(sol_high[1,:],bins=bins, normed=true,alpha=0.4, label="")
xlabel!("R")
ylabel!("P(R)")

# gluco melting peaks plot
x_0 = [0.5,0.2]
t_span = (0.0,20000.0)
p_low = (0.1,0.436,0.005)
p_mid = (0.1,0.436,0.01)
p_high = (0.1,0.436,0.02)

gluco_low = SDEProblem(gluco,noise_term_2d, x_0,t_span,p_low)
gluco_mid = SDEProblem(gluco,noise_term_2d, x_0,t_span,p_mid)
gluco_high = SDEProblem(gluco,noise_term_2d, x_0,t_span,p_high)
@time sol_low = solve(gluco_low, SROCK2();dt=0.02)
@time sol_mid = solve(gluco_mid, SROCK2();dt=0.02)
@time sol_high = solve(gluco_high, SROCK2();dt=0.02)
bins = 0.2:0.01:0.7
histogram(sol_low[1,:],bins=bins, normed=true, label="")
histogram!(sol_mid[1,:],bins=bins, normed=true,alpha=0.7, label="")
histogram!(sol_high[1,:],bins=bins, normed=true,alpha=0.4, label="")
xlabel!("R")
ylabel!("P(R)")

# normal_form melting peaks plot
x_0 = [0.0,0.1]
t_span = (0.0,20000.0)
p_low = (1, 1, 0.01, 5, 0.002)
p_mid = (1, 1, 0.01, 5, 0.02)
p_high = (1, 1, 0.01, 5, 0.04)

normal_form = @ode_def begin
    dx = x*μ - ω*y - (a*x + y*b)*(x^2+y^2)
    dy = x*ω + y*μ - (a*y - b*x)*(x^2+y^2)
end a b μ ω σ

noise_term_2d_4_var = @ode_def begin
    dx = σ
    dy = σ
end a b μ ω σ
normal_low = SDEProblem(normal_form, noise_term_2d_4_var, x_0, t_span, p_low)
normal_mid = SDEProblem(normal_form, noise_term_2d_4_var, x_0, t_span, p_mid)
normal_high = SDEProblem(normal_form, noise_term_2d_4_var, x_0,t_span, p_high)
@time soln_low = solve(normal_low, SROCK2();dt=0.02)
@time soln_mid = solve(normal_mid, SROCK2();dt=0.02)
@time soln_high = solve(normal_high, SROCK2();dt=0.02)
bins = -0.5:0.01:0.5
plot(bins,[double_gauss(2*p_high[5],0.1, x) for x in bins],color="green",linestyle=:dot,linewidth=4,label=L"f_1")
plot!(bins,[sin_ap(pi,0.05,0.1, x) for x in bins],color="blue",linestyle=:dot,linewidth=4,label=L"f_2")
histogram!(soln_low[1,:],bins=bins, normed=true,color=1,linewidth=4,grid=false, label=L"0.002")
histogram!(soln_mid[1,:],bins=bins, normed=true,color=2,linewidth=4, label=L"0.01")
histogram!(soln_high[1,:],bins=bins, normed=true,color=3,linewidth=4, label=L"0.05")
xlims!((-0.25,0.25))
plot!(xguide=L"X",guidefontsize=34)
plot!(yguide=L"P(X)",guidefontsize=30)
plot!(legend=:topright, grid=false, legendfontsize=18,
                xtickfontsize=22, ytickfontsize=22,
                margin=6Plots.mm)
plot!(size=(750,600))
savefig("C:/Users/Martiny/Dropbox/PHD/Speciale_artikel/melt.pdf")
#plot!([0.1,0.1],[0,7],linestyle = :dot,color="black",linewidth=4, label="")
#plot!([-0.1,-0.1],[0,7],linestyle = :dot,color="black",linewidth=4, label="")
#plot!([-0.1-2*p_low[5],-0.1+2*p_low[5]],[6,6],color="blue",linewidth=5, label="")
#plot!([0.1-2*p_low[5],0.1+2*p_low[5]],[5,5],color="blue",linewidth=5, label="")
#plot!([-0.1-2*p_mid[5],-0.1+2*p_mid[5]],[3.5,3.5],color="orange",linewidth=5, label="")
#plot!([0.1-2*p_mid[5],0.1+2*p_mid[5]],[3.5,3.5],color="orange",linewidth=4, label="")
#plot!([-0.1-2*p_high[5],-0.1+2*p_high[5]],[1.5,1.5],color="green",linewidth=5, label="")
#plot!([0.1-2*p_high[5],0.1+2*p_high[5]],[2,2],color="green",linewidth=3, label="")
#plot!(bins,[double_gauss(2*p_low[5],0.1, x) for x in bins])
#plot!(bins,[double_gauss(2*p_mid[5],0.1, x) for x in bins])

function double_gauss(sig, mean,x)
    forst = 1.0/2/sqrt(2*pi*sig^2)*exp(-(x+mean)^2/(2*sig^2))
    anden = 1.0/2/sqrt(2*pi*sig^2)*exp(-(x-mean)^2/(2*sig^2))
    return forst+anden
end
function sin_ap(c1,c2, mean, x)
    if abs(x) < mean
        return c1/(c2+sqrt(1-(x/mean)^2))
    else
        return 0
    end
end

μ_range = (-.1:0.02:.1)
σ_range = (0.001:0.0010:0.010)
data, data_low, data_up = Funcs.raster_scan_4var(x_0, t_span, normal_form, noise_term_2d_4_var, μ_range, σ_range, 3, 60)
