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
using ColorSchemes

#brusselator
brusselator = @ode_def begin
   dx = a + y*x^2 - b*x - x
   dy = b * x - y * x^2
end a b σ

noise_term_2d = @ode_def begin
   dx = σ
   dy = σ
end a b σ

xrange = (0.0:0.015:5.0)
yrange = (0.0:0.015:5.0)
meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x)))

x, y = meshgrid(0.5:0.11:2, 1:0.11:3)
u = @. 1 + y*x^2 - 2.1*x - x
v= @. 2.1 * x - y * x^2
scale = 0.2


x_0 = [1.0,1.0]
t_span = (0.0,100.0)
p=(1, 2.1, 0.00)
ndata=10
nbins =60
prob_sde_brusselator = SDEProblem(brusselator,noise_term_2d, x_0,t_span,p)
@time sol = solve(prob_sde_brusselator, SROCK2();dt=0.01)

x_0 = [1.0,1.0]
t_span = (0.0,100.0)
p=(1, 2.1, 0.03)
ndata=10
nbins =60
prob_sde_brusselator = SDEProblem(brusselator,noise_term_2d, x_0,t_span,p)
@time soln = solve(prob_sde_brusselator, SROCK2();dt=0.01)

x_0 = [1.0,1.0]
t_span = (0.0,100.0)
p=(1, 1.98, 0.02)
ndata=10
nbins =60
prob_sde_brusselator = SDEProblem(brusselator,noise_term_2d, x_0,t_span,p)
@time sol_ndo = solve(prob_sde_brusselator, SROCK2();dt=0.001)
plot(layout=(1,1))
quiver!(x, y, quiver=(u*scale, v*scale),subplot=1)
xlims!((0.5,1.7))
ylims!((1.3,2.7))
a = Array(sol)
an = Array(soln)
ando = Array(sol_ndo)
plot!(an[1,7000:end],an[2,7000:end],linewidth=3,label="")
plot!(a[1,7000:end],a[2,7000:end],linewidth=5,label="")
plot!(ando[1,70000:50:end], ando[2,70000:50:end],linewidth=5,label="")
plot!([1.1,1.12],[2.33,2.38], line = ( :arrow, 4, :black),label="")
plot!([1.45,1.55],[1.63,1.52], line = ( :arrow, 4, :black),label="")
scatter!((1,1.98),markersize = 10,color=:blue,label="",ticks=false)
plot!(legend=false, grid=false, legendfontsize=16,
                xlabelfontsixe=30,
                xtickfontsize=26, ytickfontsize=26, margin=6Plots.mm)
plot!(xguide="X",guidefontsize=24)
plot!(yguide="Y",guidefontsize=24)
plot!(size=(750,500))
savefig("C:/Users/Martiny/Dropbox/PHD/Speciale_artikel/intro_limit.pdf")
pyplot()
x, y = meshgrid(0.5:0.03:2, 1:0.03:3)
u = @. 1 + y*x^2 - 1.5*x - x
v= @. 1.5 * x - y * x^2
scale = 0.05


x_0 = [1.0,1.0]
t_span = (0.0,100.0)
p=(1, 1.5, 0.01)
ndata=10
nbins =60
prob_sde_brusselator = SDEProblem(brusselator,noise_term_2d, x_0,t_span,p)
@time sol = solve(prob_sde_brusselator, SROCK2();dt=0.01)

quiver(x, y, quiver=(u*scale, v*scale))
xlims!((0.7,1.3))
ylims!((1.3,1.7))
a = Array(sol)
plot!(a[1,6000:end],a[2,6000:end],label="")
plot!([1.1,1.13],[2.3,2.4], line = ( :arrow, 4, :red),label="")
plot!([1.45,1.55],[1.6,1.47], line = ( :arrow, 4, :red),label="")
xlabel!("x")
ylabel!("y")
savefig("C:/Users/emils/Dropbox/PHD/Speciale_artikel/intro_limit.pdf")
