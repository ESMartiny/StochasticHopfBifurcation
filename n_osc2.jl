include("./Funcs.jl")
using .Funcs

using LinearAlgebra
using Plots
using DifferentialEquations
using ParameterizedFunctions
#using FFTW
using LsqFit
using StatsBase
using JLD
using LaTeXStrings

# brusselator
brusselator = @ode_def begin
    dx = a + y*x^2 - b*x - x
    dy = b * x - y * x^2
end a b σ
gluco = @ode_def begin
  dx = -x + a*y + x^2*y
  dy = b - a*y - x^2*y
end a b σ

noise_term_2d = @ode_def begin
    dx = σ
    dy = σ
end a b σ

x_0 = [0.5,0.5]
t_span = (0.0,150.0)
function gluco_conversion(λ)
    a = 0.1
    b2 = 0.5*(1 - 2*a - 2*λ - sqrt(4*λ^2 + 1 - 8*a - 4*λ))
    return sqrt(b2)
end
p=(0.1, gluco_conversion(0.01), 0.02)
nbins = 100
prob_sde_gluco = SDEProblem(gluco,noise_term_2d, x_0,t_span,p)
@time sol = solve(prob_sde_gluco, SROCK2();dt=0.05)
plot(sol,vars=(1),color="red", label="")

function get_param(d,len,dt_inv,b)
    rs = Funcs.time_emb_r(vec(d[1,1:len*dt_inv]), b)
    rs_sorted = sort(rs)
    ndata = 1.1
    l = floor(Int64,length(rs_sorted)/1.1)
    h = fit(Histogram, rs_sorted[1:l], nbins=nbins)
    toppoint = argmax(h.weights)
    ls = floor(Int64,toppoint/ndata)
    empi = h.weights[1:ls]
    @. model(x, p) = p[1] * x + p[2]*x^2
    #print(length(empi))
    a = (empi[end]/(length(empi)-1))
    xdata = 0:length(empi)-1
    p0 = [a,0]
    lb = [0, -Inf]
    ub = [length(empi), Inf]
    #f = curve_fit(model2, xdata, empi, p0,lower=lb, upper=ub)
    f = curve_fit(model, xdata, empi, p0)
    return f.param[2]
end
function gluco_conversion(λ)
    a = 0.1
    b2 = 0.5*(1 - 2*a - 2*λ - sqrt(4*λ^2 + 1 - 8*a - 4*λ))
    return sqrt(b2)
end
gluco_conversion(-0.01)
b_range = [-0.03,-0.02,-0.01,0.01,0.02,0.03]
b_range = [gluco_conversion(b) for b in b_range]
length_range = [n for n in 100:100:1000]
append!(length_range,[n for n in 1000:500:10000])
data_all = Dict()
for b in b_range
    data=[]
    x_0 = [1.0,1.0]
    t_span = (0.0,10000.0)
    p=(0.1, b, 0.04)
    nbins = 100
    prob_sde_gluco = SDEProblem(gluco,noise_term_2d, x_0,t_span,p)
    nIter = 50
    for n in range(1,stop=nIter)
        sol = solve(prob_sde_gluco, SROCK2();dt=0.05)
        append!(data,[sol])
        print(n)
    end
    ana_data = zeros(length(data),length(length_range))
    for (i,d) in enumerate(data)
        for (j,len) in enumerate(length_range)
            p = get_param(d,len,20,b)
            if p > 0
                ana_data[i,j] = 1
            end
        end
    end
    data_all[b]=ana_data
end
fnt = Plots.font("Helvetica", 120.0)
legfnt = Plots.font("Helvetica", 2.0)
plot(legend=true, grid=false, legendfontsize=18,
                xtickfontsize=22, ytickfontsize=22,
                margin=6Plots.mm)
b = b_range[6]
a = sum(data_all[b],dims=1)./50
a = [a[1,n] for n in range(1,length(length_range))]
p1=plot!(length_range./10,a,linewidth=4,label="0.03",legend=:right)
plot!(xguide=L"\# Osc ",guidefontsize=26)
plot!(yguide=L"P(DC)",guidefontsize=26)
plot!(legend=:right, grid=false, legendfontsize=20,
                xtickfontsize=22, ytickfontsize=22,
                margin=6Plots.mm)
plot!(size=(750,600))
savefig(p1,"C:/Users/Martiny/Dropbox/PHD/Speciale_artikel/noscgluco_highnoise.pdf")


x_0 = [0.5,0.5]
t_span = (0.0,150.0)
function gluco_conversion(λ)
    a = 0.1
    b2 = 0.5*(1 - 2*a - 2*λ - sqrt(4*λ^2 + 1 - 8*a - 4*λ))
    return sqrt(b2)
end

p=(0.1, gluco_conversion(-0.01), 0.02)
nbins = 100
t_span = (0.0,150.0)
prob_sde_gluco_ndo = SDEProblem(gluco,noise_term_2d, x_0,t_span,p)
@time sol = solve(prob_sde_gluco_ndo, SROCK2();dt=0.05)
plot(sol,vars=(1),color=3,linewidth=3, label="")
plot!(xguide=L"t",guidefontsize=34)
plot!(yguide=L"X",guidefontsize=30)
yticks!([0.2,0.4,0.6])
xticks!([0,75,150])
plot!(legend=:right, grid=false, legendfontsize=16,
                xlabelfontsixe=22,
                xtickfontsize=26, ytickfontsize=26, margin=6Plots.mm)

plot!(size=(750,500))
savefig("C:/Users/Martiny/Dropbox/PHD/Speciale_artikel/noscgluco_x(t)ndo.pdf")
p=(0.1, gluco_conversion(0.02), 0.02)
prob_sde_gluco_lco = SDEProblem(gluco,noise_term_2d, x_0,t_span,p)
@time sol = solve(prob_sde_gluco_lco, SROCK2();dt=0.05)
plot(sol,vars=(1),color=5,linewidth=3, label="")
plot!(xguide=L"t",guidefontsize=34)
plot!(yguide=L"X",guidefontsize=30)
yticks!([0.2,0.5,0.8])
xticks!([0,75,150])
plot!(legend=:right, grid=false, legendfontsize=16,
                xtickfontsize=22, ytickfontsize=22,
                margin=6Plots.mm)
plot!(size=(750,500))
savefig("C:/Users/Martiny/Dropbox/PHD/Speciale_artikel/noscgluco_x(t)lco.pdf")


p=(0.1, gluco_conversion(-0.01), 0.04)
nbins = 100
t_span = (0.0,150.0)
prob_sde_gluco_ndo = SDEProblem(gluco,noise_term_2d, x_0,t_span,p)
@time sol = solve(prob_sde_gluco_ndo, SROCK2();dt=0.05)
plot(sol,vars=(1),color=3,linewidth=3, label="")
yticks!([0.2,0.4,0.6])
xticks!([0,75,150])
plot!(xguide=L"t",guidefontsize=34)
plot!(yguide=L"X",guidefontsize=30)
plot!(legend=:right, grid=false, legendfontsize=16,
                xlabelfontsixe=22,
                xtickfontsize=22, ytickfontsize=22, margin=6Plots.mm)
plot!(size=(750,500))
savefig("C:/Users/Martiny/Dropbox/PHD/Speciale_artikel/noscgluco_x(t)ndo_highnoise.pdf")
p=(0.1, gluco_conversion(0.02), 0.04)
prob_sde_gluco_lco = SDEProblem(gluco,noise_term_2d, x_0,t_span,p)
@time sol = solve(prob_sde_gluco_lco, SROCK2();dt=0.05)
plot(sol,vars=(1),color=5,linewidth=3, label="")
plot!(xguide=L"t",guidefontsize=34)
plot!(yguide=L"X",guidefontsize=30)
yticks!([0.2,0.5,0.8])
xticks!([0,75,150])
plot!(legend=:right, grid=false, legendfontsize=16,
                xtickfontsize=22, ytickfontsize=22,
                margin=6Plots.mm)
plot!(size=(750,500))
savefig("C:/Users/Martiny/Dropbox/PHD/Speciale_artikel/noscgluco_x(t)lco_highnoise.pdf")
