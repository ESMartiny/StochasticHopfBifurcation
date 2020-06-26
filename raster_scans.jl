include("./Funcs.jl")
using .Funcs

using LinearAlgebra
using Plots
using DifferentialEquations
using ParameterizedFunctions
using FFTW
using LsqFit
using StatsBase
using JLD
using LaTeXStrings


gluco = @ode_def begin
  dx = -x + a*y + x^2*y
  dy = b - a*y - x^2*y
end a b σ

noise_term_2d = @ode_def begin
    dx = σ
    dy = σ
end a b σ

noise_term_2d_multi = @ode_def begin
    dx = σ * sqrt(x)
    dy = σ * sqrt(y)
end a b σ
b_range = (-0.03:0.001:0.03)
b_convo = map(gluco_conversion, b_range)
σ_range = (0.002:0.0020:0.050)
length(σ_range)
x_0 = [1.0,1.0]
t_span = (0.0,25000.0)
@time data = Funcs.raster_scan_no_fit_gluco(x_0, t_span, gluco, noise_term_2d, b_convo, σ_range, 3, 200)
save("gluco_raster.jld", "gluco_raster", data)
b_range = (-0.03:0.0015:0.03)
b_convo = map(gluco_conversion, b_range)
σ_range = (0.002:0.0015:0.050)
@time data = Funcs.raster_scan_no_fit_gluco(x_0, t_span, gluco, noise_term_2d_multi, b_convo, σ_range, 3, 200)
save("gluco_raster_multi.jld", "gluco_raster_multi", data)
# Normal form
normal_form = @ode_def begin
    dx = x*μ - ω*y - (a*x + y*b)*(x^2+y^2)
    dy = x*ω + y*μ - (a*y - b*x)*(x^2+y^2)
end a b μ ω σ
noise_term_2d_4_var = @ode_def begin
    dx = σ
    dy = σ
end a b μ ω σ
noise_term_2d_4_var_m = @ode_def begin
    dx = σ * sqrt(x)
    dy = σ * sqrt(y)
end a b μ ω σ

b_range = (-0.03:0.001:0.03)
σ_range = (0.002:0.0010:0.050)
x_0 = [0.0,0.1]
t_span = (0.0,50000.0)
@time data = Funcs.raster_scan_no_fit_normal(x_0, t_span, normal_form, noise_term_2d_4_var, b_range, σ_range, 3, 200)
save("normal_raster.jld", "normal_raster", data)
t_span = (0.0,50000.0)
@time data = Funcs.raster_scan_no_fit_normal(x_0, t_span, normal_form, noise_term_2d_4_var_m, b_range, σ_range, 3, 200)
save("normal_raster_m.jld", "normal_raster_m", data)


#brusselator
function brusselator_conversion(λ)
    a = 1
    return 2*λ + 1 + a^2
end

brusselator = @ode_def begin
    dx = a + y*x^2 - b*x - x
    dy = b * x - y * x^2
end a b σ
b_range = (-0.03:0.001:0.03)
b_range = (-0.04:0.0015:0.04)
b_convo = map(brusselator_conversion, b_range)
σ_range = (0.002:0.0020:0.050)
σ_range = (0.002:0.001:0.050)
x_0 = [1.0,1.0]
t_span = (0.0,30000.0)
@time data = Funcs.raster_scan_no_fit(x_0, t_span, brusselator, noise_term_2d, b_convo, σ_range, 3, 200)
save("brusselator_raster.jld", "brusselator_raster", data)
b_range = (-0.03:0.0015:0.03)
b_convo = map(brusselator_conversion, b_range)
σ_range = (0.002:0.0015:0.050)
@time data = Funcs.raster_scan_no_fit_brusselator(x_0, t_span, brusselator, noise_term_2d_multi, b_convo, σ_range, 3, 200)
save("brusselator_raster_multi.jld", "brusselator_raster_multi", data)
plot(data[1,1])
#chlorine
function chlorine_conversion(λ)
    a = 10
    b = -10*λ/a - 2*a*λ/5 + 3*a/5 - 25/a
    return b
end

chlorine = @ode_def begin
    dx = a - x - 4*x*y/(1+x^2)
    dy = b*x*(1-y/(1+x^2))
end a b

b_range = (-0.03:0.0015:0.03)
b_convo = map(chlorine_conversion, b_range)
σ_range = (0.002:0.0015:0.050)
length(σ_range)
x_0 = [2.0,5.0]
t_span = (0.0,5000.0)
@time data_chlor = Funcs.raster_scan_no_fit(x_0, t_span, chlorine, noise_term_2d, b_convo, σ_range, 3, 200)
save("chlorine_raster.jld", "chlorine_raster", data_chlor)
b_range = (-0.03:0.0015:0.03)
b_convo = map(chlorine_conversion, b_range)
σ_range = (0.002:0.0015:0.050)
@time data_chlor = Funcs.raster_scan_no_fit(x_0, t_span, chlorine, noise_term_2d_multi, b_convo, σ_range, 3, 200)
save("chlorine_raster_multi.jld", "chlorine_multi_raster", data_chlor)






data_chlor = load("chlorine_raster.jld","chlorine_raster")
kolmo = Funcs.raster_of_sim_smart(data_chlor, b_convo, σ_range, 1.1, 60, "kolmo")
contourf(σ_range, b_range, kolmo)
second, other = Funcs.raster_of_sim_smart(data_chlor, b_convo, σ_range, 1., 60, "second")
second, other = Funcs.raster_of_sim_smart(data_chlor, b_convo, σ_range, 1, 1.05, "second_1degree")
errormat = zeros(length(b_range),length(σ_range))
for i in range(1,length(b_range))
    for j in range(1,length(σ_range))
        errormat[i,j] = stderror(other[i,j])[2]
    end
end
clamp.(second./errormat,-25.0,25.0)
contourf(σ_range, b_range, clamp.(second./errormat,-25,25),color=:balance, levels=[-25.1,-17,-10,0,10,17,25.1])
contourf(σ_range, b_range, second,color=:balance, levels=[-25.1,-17,-10,0,10,17,25.1])
contourf(σ_range, b_range, clamp.(second./errormat,-25,25),color=:curl)

plot(data_chlor[3,2])
x_0 = [2.0,16.0]
b = chlorine_conversion(0.0)
t_span = (0.0,100.0)
p = (10,b,0.1)
prob_sde_chlor= SDEProblem(chlorine, noise_term_2d, x_0, t_span, p)
@time sol = solve(prob_sde_chlor, SROCK2(); dt=0.005)
plot(sol)


x_0 = [4.0,16.0]
chlorine_conversion(0.01)
t_span = (0.0,50.0)
p = (0.1,0.44,0.01)
prob_sde_gluco= SDEProblem(gluco, noise_term_2d, x_0, t_span, p)
@time sol = solve(prob_sde_gluco, SROCK2(); dt=0.01)
plot(sol)
fi = fit_second_order_1var_plot(sol,p[2],15,50)
gluco_conversion(0.1)
function gluco_conversion(λ)
    a = 0.1
    b2 = 0.5*(1 - 2*a - 2*λ - sqrt(4*λ^2 + 1 - 8*a - 4*λ))
    return sqrt(b2)
end

gluco_conversion(0.01)

function chlorine_conversion(λ)
    a = 0.5
    b = -10λ/a - 2a*λ/5 + 3a/5 - 25/a
    return b
end

#glucolysis raster fig
data_gluco = load("C:/Users/emils/Dropbox/PHD/Speciale_artikel/gluco_raster.jld","gluco_raster")
b_range = (-0.03:0.001:0.03)
b_convo = map(gluco_conversion, b_range)
σ_range = (0.002:0.0020:0.050)
kolmo = Funcs.raster_of_sim_smart(data_gluco, b_convo, σ_range, 1.1, 60, "kolmo")
contourf(σ_range, b_range, kolmo,
                            levels=[0.0,0.025,0.05,0.10,0.15,0.2,0.3,0.5],
                            xtickfont = font(10),
                            ytickfont = font(10))
#plot!(σ_range,(σ_range/2).^(2/3))
plot!(σ_range[1:21],σ_range[1:21]/sqrt(2),linewidth=4,label="")
plot!(xguide=L"\sigma", guidefontsize=30)
plot!(yguide=L"\lambda", guidefontsize=30)
#yaxis!([-0.03,0.03])
scatter!([σ_range[7],σ_range[15],σ_range[15]], [b_range[20],b_range[45],b_range[55]],
                    label="",
                    markersize=14,
                    markercolor= :white)
scatter!([σ_range[7],σ_range[15],σ_range[15]],[b_range[20],b_range[45],b_range[55]],
                    label="",
                    markersize=10,
                    markercolor= [:red, :green,:blue])
                    #[[:red :white],:green,:blue])

b_convo[10]

plot!(legend=:right, grid=false, legendfontsize=16,
                xtickfontsize=22, ytickfontsize=22,
                margin=6Plots.mm)

plot!(size=(750,500))
#Plots.scalefontsizes(1.3)
savefig("C:/Users/emils/Dropbox/PHD/Speciale_artikel/gluco_raster.pdf")
second, other = Funcs.raster_of_sim_smart(data_gluco, b_convo, σ_range, 1.1, 60, "second")
errormat = zeros(length(b_range),length(σ_range))
for i in range(1,length(b_range))
    for j in range(1,length(σ_range))
        errormat[i,j] = stderror(other[i,j])[2]
    end
end
clamp.(second./errormat,-25.0,25.0)
contourf(σ_range, b_range, clamp.(second./errormat,-25,25),color=:balance,
                        levels=[-25.1,-17,-10,0,10,17,25.1],
                        xtickfont = font(10),
                        ytickfont = font(10))

xs = [0.02,0.02,0.02,0.02,0.02,0.02,0.04,0.04,0.04,0.04,0.04,0.04]
ys = [-0.03,-0.02,-0.01,0.01,0.02,0.03,-0.03,-0.02,-0.01,0.01,0.02,0.03]
scatter!(xs,ys,
                    label="",
                    markersize=14,
                    markercolor=:white)
scatter!(xs,ys,
                    label="",
                    markersize=10,
                    markercolor=[1,2,3,4,5,6,1,2,3,4,5,6])
plot!(xguide=L"\sigma", guidefontsize=30)
plot!(yguide=L"\lambda", guidefontsize=30)
plot!(legend=:right, grid=false, legendfontsize=16,
                    xtickfontsize=22, ytickfontsize=22,
                    margin=6Plots.mm)
plot!(size=(750,500))
savefig("C:/Users/emils/Dropbox/PHD/Speciale_artikel/gluco_raster_second.pdf")
#examples
h = data_gluco[20,7]
#Plots.scalefontsizes(1.3)
plot(normalize(h),color=:red, label="")
plot!(xguide=L"R", guidefontsize=30)
plot!(yguide=L"P(R)", guidefontsize=30)

ndata=1
binso = h.edges
bins = (0.0:0.001:0.05)

plot!(bins, [b*200 for b in bins],color="green",linestyle=:dot,linewidth=14,label="")
plot!(legend=:right, grid=false, legendfontsize=16,
                    xtickfontsize=22, ytickfontsize=22,
                    margin=6Plots.mm)
plot!(size=(750,500))

toppoint = argmax(h.weights)
l = floor(Int64,toppoint/ndata)
empi = h.weights[1:l]
@. model(x, p) = p[1] * x + p[2]*x^2
a = (empi[end]/(length(empi)-1))
xdata = 0:length(empi)-1
typeof(binso[1])
p0 = [a,0]
lb = [0, -Inf]
ub = [length(empi), Inf]
#f = curve_fit(model2, xdata, empi, p0,lower=lb, upper=ub)
f = curve_fit(model, xdata, empi, p0)
m = model(binso[1], f.param)
m = m./m[end]*binso[end]*27
m = prepend!(m,0)
length(binso[1])
#plot!(bins,m[1:length(bins)]*2,linewidth=3,color="black",labels="")
savefig("C:/Users/emils/Dropbox/PHD/Speciale_artikel/hist_ex1.pdf")
#examples
h = data_gluco[45,15]
#Plots.scalefontsizes(1.3)
plot(normalize(h),color=:green, label="")
plot!(xguide=L"R", guidefontsize=30)
plot!(yguide=L"P(R)", guidefontsize=30)
ndata=1.2
binso = h.edges
toppoint = argmax(h.weights)
l = floor(Int64,toppoint/ndata)
empi = h.weights[1:l]
@. model(x, p) = p[1] * x + p[2]*x^2
a = (empi[end]/(length(empi)-1))
xdata = 0:length(empi)-1
typeof(binso[1])
p0 = [a,0]
lb = [0, -Inf]
ub = [length(empi), Inf]
#f = curve_fit(model2, xdata, empi, p0,lower=lb, upper=ub)
f = curve_fit(model, xdata, empi, p0)
m = model(range(1,85), f.param)
m = m./m[end]*5.45

length(binso[1])
plot!(binso[1][1:85],m[1:85],linewidth=14,linestyle=:dot,color="red",labels="")
plot!(legend=:right, grid=false, legendfontsize=16,
                    xtickfontsize=22, ytickfontsize=22,
                    margin=6Plots.mm)
plot!(size=(750,500))
savefig("C:/Users/emils/Dropbox/PHD/Speciale_artikel/hist_ex2.pdf")
h = data_gluco[55,15]
#Plots.scalefontsizes(1.3)
plot(normalize(h),color=:blue, label="")
plot!(xguide=L"R", guidefontsize=30)
plot!(yguide=L"P(R)", guidefontsize=30)
ndata=1.2
binso = h.edges
toppoint = argmax(h.weights)
l = floor(Int64,toppoint/ndata)
empi = h.weights[1:l]
@. model(x, p) = p[1] * x + p[2]*x^2
a = (empi[end]/(length(empi)-1))
xdata = 0:length(empi)-1
typeof(binso[1])
p0 = [a,0]
lb = [0, -Inf]
ub = [length(empi), Inf]
#f = curve_fit(model2, xdata, empi, p0,lower=lb, upper=ub)
f = curve_fit(model, xdata, empi, p0)
m = model(range(1,100), f.param)
m = m./m[end]*5.3

length(binso[1])
plot!(binso[1][1:100],m[1:100],linewidth=14,linestyle=:dot,color="red",labels="")
plot!(legend=:right, grid=false, legendfontsize=16,
                    xtickfontsize=22, ytickfontsize=22,
                    margin=6Plots.mm)
plot!(size=(750,500))
savefig("C:/Users/emils/Dropbox/PHD/Speciale_artikel/hist_ex3.pdf")


# gluco multiplicative noise plot
data_gluco_m = load("C:/Users/emils/Dropbox/PHD/Speciale_artikel/gluco_raster_multi.jld","gluco_raster_multi")
b_range = (-0.03:0.0015:0.03)
b_convo = map(gluco_conversion, b_range)
σ_range = (0.002:0.0015:0.050)
kolmo = Funcs.raster_of_sim_smart(data_gluco_m, b_convo, σ_range, 1.1, 60, "kolmo")
contourf(σ_range, b_range, kolmo,levels=[0.0,0.025,0.05,0.10,0.15,0.2,0.3,0.5])
plot!(legend=:right, grid=false, legendfontsize=16,
                    xtickfontsize=20, ytickfontsize=20,
                    margin=6Plots.mm)
plot!([0.01],[0.0],label="")
plot!(xguide=L"\sigma", guidefontsize=30)
plot!(yguide=L"\lambda", guidefontsize=30)
plot!(size=(750,500))
savefig("C:/Users/emils/Dropbox/PHD/Speciale_artikel/gluco_raster_multi.pdf")
second, other = Funcs.raster_of_sim_smart(data_gluco_m, b_convo, σ_range, 1.1, 60, "second")
errormat = zeros(length(b_range),length(σ_range))
for i in range(1,length(b_range))
    for j in range(1,length(σ_range))
        errormat[i,j] = stderror(other[i,j])[2]
    end
end
clamp.(second./errormat,-25.0,25.0)
contourf(σ_range, b_range, clamp.(second./errormat,-25,25),color=:balance, levels=[-25.1,-17,-10,0,10,17,25.1])

plot!([0.01],[0.0],label="")
plot!(xguide=L"\sigma", guidefontsize=30)
plot!(yguide=L"\lambda", guidefontsize=30)
plot!(legend=:right, grid=false, legendfontsize=16,
                    xtickfontsize=20, ytickfontsize=20,
                    margin=6Plots.mm)
plot!(size=(750,500))
savefig("C:/Users/emils/Dropbox/PHD/Speciale_artikel/gluco_raster_multi_second.pdf")

#normal form raster fig
data_normal = load("normal_raster.jld","normal_raster")
b_range = (-0.03:0.001:0.03)
#b_convo = map(gluco_conversion, b_range)
σ_range = (0.002:0.0020:0.050)
kolmo = Funcs.raster_of_sim_smart(data_normal, b_range, σ_range, 1.1, 60, "kolmo")
contourf(σ_range, b_range, kolmo,levels=[0.0,0.025,0.05,0.10,0.15,0.2,0.3,0.5])
plot!([0.01],[0.0],label="")
plot!(xguide=L"\sigma", guidefontsize=30)
plot!(yguide=L"\lambda", guidefontsize=30)
plot!(legend=:right, grid=false, legendfontsize=16,
                    xtickfontsize=20, ytickfontsize=20,
                    margin=6Plots.mm)
plot!(size=(750,500))
savefig("C:/Users/emils/Dropbox/PHD/Speciale_artikel/normal_raster.pdf")
second, other = Funcs.raster_of_sim_smart(data_normal, b_range, σ_range, 1.1, 60, "second")
errormat = zeros(length(b_range),length(σ_range))
for i in range(1,length(b_range))
    for j in range(1,length(σ_range))
        errormat[i,j] = stderror(other[i,j])[2]
    end
end
clamp.(second./errormat,-25.0,25.0)
contourf(σ_range, b_range, clamp.(second./errormat,-25,25),color=:balance, levels=[-25.1,-17,-10,0,10,17,25.1])
plot!([0.01],[0.0],label="")
plot!(xguide=L"\sigma", guidefontsize=30)
plot!(yguide=L"\lambda", guidefontsize=30)
plot!(legend=:right, grid=false, legendfontsize=16,
                    xtickfontsize=20, ytickfontsize=20,
                    margin=6Plots.mm)
plot!(size=(750,500))
savefig("C:/Users/emils/Dropbox/PHD/Speciale_artikel/normal_raster_second.pdf")

#brusselator raster fig
data_brusselator = load("brusselator_raster.jld","brusselator_raster")
b_range = (-0.04:0.0015:0.04)
b_convo = map(brusselator_conversion, b_range)
σ_range = (0.002:0.001:0.050)
kolmo = Funcs.raster_of_sim_smart(data_brusselator, b_convo, σ_range, 1.1, 60, "kolmo")
contourf(σ_range, b_range, kolmo,levels=[0.0,0.025,0.05,0.10,0.15,0.2,0.3,0.5])
xlabel!(L"\sigma")
ylabel!(L"\lambda")
savefig("C:/Users/emils/Dropbox/PHD/Speciale_artikel/brusselator_raster.pdf")
second, other = Funcs.raster_of_sim_smart(data_brusselator, b_convo, σ_range, 1.1, 60, "second")
errormat = zeros(length(b_range),length(σ_range))
for i in range(1,length(b_range))
    for j in range(1,length(σ_range))
        errormat[i,j] = stderror(other[i,j])[2]
    end
end
clamp.(second./errormat,-25.0,25.0)
contourf(σ_range, b_range, clamp.(second./errormat,-25,25),color=:balance, levels=[-25.1,-17,-10,0,10,17,25.1])
xlabel!(L"\sigma")
ylabel!(L"\lambda")
savefig("C:/Users/emils/Dropbox/PHD/Speciale_artikel/brusselator_raster_second.pdf")
#brusselator multi
data_brusselator = load("brusselator_raster_multi.jld","brusselator_raster_multi")
b_range = (-0.03:0.0015:0.03)
b_convo = map(brusselator_conversion, b_range)
σ_range = (0.002:0.0015:0.050)
kolmo = Funcs.raster_of_sim_smart(data_brusselator, b_convo, σ_range, 1.1, 60, "kolmo")
contourf(σ_range, b_range, kolmo,levels=[0.0,0.025,0.05,0.10,0.15,0.2,0.3,0.5])
ylabel!(L"\lambda")
xlabel!(L"\sigma")
savefig("C:/Users/emils/Dropbox/PHD/Speciale_artikel/brusselator_raster_multi.pdf")
second, other = Funcs.raster_of_sim_smart(data_brusselator, b_convo, σ_range, 1.1, 60, "second")
errormat = zeros(length(b_range),length(σ_range))
for i in range(1,length(b_range))
    for j in range(1,length(σ_range))
        errormat[i,j] = stderror(other[i,j])[2]
    end
end
clamp.(second./errormat,-25.0,25.0)
contourf(σ_range, b_range, clamp.(second./errormat,-25,25),color=:balance, levels=[-25.1,-17,-10,0,10,17,25.1])
xlabel!(L"\sigma")
ylabel!(L"\lambda")
savefig("C:/Users/emils/Dropbox/PHD/Speciale_artikel/brusselator_raster_multi_second.pdf")

#chlorine raster fig
data_chlorine= load("chlorine_raster.jld","chlorine_raster")
b_range = (-0.03:0.0015:0.03)
b_convo = map(chlorine_conversion, b_range)
σ_range = (0.002:0.0015:0.050)
kolmo = Funcs.raster_of_sim_smart(data_chlorine, b_convo, σ_range, 1.1, 60, "kolmo")
contourf(σ_range, b_range, kolmo,levels=[0.0,0.025,0.05,0.10,0.15,0.2,0.3,0.5,1])
xlabel!(L"\sigma")
ylabel!(L"\lambda")
savefig("C:/Users/emils/Dropbox/PHD/Speciale_artikel/chlorine_raster.pdf")
second, other = Funcs.raster_of_sim_smart(data_chlorine, b_convo, σ_range, 1., 60, "second")
errormat = zeros(length(b_range),length(σ_range))
for i in range(1,length(b_range))
    for j in range(1,length(σ_range))
        errormat[i,j] = stderror(other[i,j])[2]
    end
end
clamp.(second./errormat,-25.0,25.0)
contourf(σ_range, b_range, clamp.(second./errormat,-25,25),color=:balance, levels=[-25.1,-17,-10,-3,3,10,17,25.1])
xlabel!(L"\sigma")
ylabel!(L"\lambda")
savefig("C:/Users/emils/Dropbox/PHD/Speciale_artikel/chlorine_raster_second.pdf")
# chlorine multi
data_chlorine_m= load("chlorine_raster_multi.jld","chlorine_multi_raster")
b_range = (-0.03:0.0015:0.03)
b_convo = map(chlorine_conversion, b_range)
σ_range = (0.002:0.0015:0.050)
kolmo = Funcs.raster_of_sim_smart(data_chlorine_m, b_convo, σ_range, 1.2, 60, "kolmo")
contourf(σ_range, b_range, kolmo,levels=[0.0,0.025,0.05,0.10,0.15,0.2,0.3,0.5,1])

xlabel!(L"\sigma")
ylabel!(L"\lambda")
savefig("C:/Users/emils/Dropbox/PHD/Speciale_artikel/chlorine_raster_multi.pdf")
second, other = Funcs.raster_of_sim_smart(data_chlorine_m, b_convo, σ_range, 1., 60, "second")
plot!(σ_range,4*σ_range.^2)
errormat = zeros(length(b_range),length(σ_range))
for i in range(1,length(b_range))
    for j in range(1,length(σ_range))
        errormat[i,j] = stderror(other[i,j])[2]
    end
end
clamp.(second./errormat,-25.0,25.0)
contourf(σ_range, b_range, clamp.(second./errormat,-25,25),color=:balance, levels=[-25.1,-17,-10,-3,3,10,17,25.1])
xlabel!(L"\sigma")
ylabel!(L"\lambda")
savefig("C:/Users/emils/Dropbox/PHD/Speciale_artikel/chlorine_raster_second_multi.pdf")


kolmo = Funcs.raster_of_sim_smart(data, b_range, σ_range, 1.1, 60, "kolmo")
contourf(σ_range, b_range, kolmo,levels=[0.0,0.025,0.05,0.10,0.15,0.2,0.3,0.5])
plot!(σ_range,σ_range/2,linewidth=3,color="green",linestyle=:dash,labels="")
second, other = Funcs.raster_of_sim_smart(data, b_range, σ_range, 1.2, 60, "second")
errormat = zeros(length(b_range),length(σ_range))
for i in range(1,length(b_range))
    for j in range(1,length(σ_range))
        errormat[i,j] = stderror(other[i,j])[2]
    end
end
clamp.(second./errormat,-25.0,25.0)
contourf(σ_range, b_range, clamp.(second./errormat,-25,25),levels=[-25.1,-17,-10,-2,2,10,17,25.1],color=:balance)
plot(data[1,1])

plot(data_chlorine[20,10])
