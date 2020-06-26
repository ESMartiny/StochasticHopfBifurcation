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
function brusselator_conversoin(λ)
    a = 1
    return 2*λ + 1 + a^2
end
brusselator_conversoin(0.01)
x_0 = [1.0,1.0]
t_span = (0.0,6000.0)
p=(1, 2.04, 0.02)
nbins = 100
prob_sde_brusselator = SDEProblem(brusselator,noise_term_2d, x_0,t_span,p)
@time sol = solve(prob_sde_brusselator, SROCK2();dt=0.01)
plot(sol,vars=(1),color="red", label="")

data = []
nIter = 50
for n in range(1,stop=nIter)
    sol = solve(prob_sde_brusselator, SROCK2();dt=0.01)
    append!(data,[sol])
    print(n)
end
data
save("n_osc1.jld", "n_osc", ana_data1)
length_range = (100:50:800)
#p=(1, 2.06, 0.02)
ana_data1 = zeros(length(data),length(length_range))

function get_param(d,len)
    rs = Funcs.time_emb_r(vec(d[1,1:len*100]), 1)
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
ana_data2 = zeros(length(data),length(length_range))
for (i,d) in enumerate(data)
    for (j,len) in enumerate(length_range)
        p = get_param(d,len)
        if p > 0
            ana_data2[i,j] = 1
        end
    end
    print(i)
end
a = (sum(ana_data2,dims=1))
a = [a[1,n] for n in range(1,length(length_range))]
plot(length_range./3,a./50)
xlabel!(L"N Oscillations")
ylabel!(L"<T>")

b_range = (1.94,1.96,1.98,2.02,2.04,2.06)
length_range = [n for n in 100:100:1000]
append!(length_range,[n for n in 1000:250:6000])
data_all = Dict()
for b in b_range
    data=[]
    x_0 = [1.0,1.0]
    t_span = (0.0,6000.0)
    p=(1, b, 0.02)
    nbins = 100
    prob_sde_brusselator = SDEProblem(brusselator,noise_term_2d, x_0,t_span,p)
    nIter = 50
    for n in range(1,stop=nIter)
        sol = solve(prob_sde_brusselator, SROCK2();dt=0.01)
        append!(data,[sol])
        print(n)
    end
    ana_data = zeros(length(data),length(length_range))
    for (i,d) in enumerate(data)
        for (j,len) in enumerate(length_range)
            p = get_param(d,len)
            if p > 0
                ana_data[i,j] = 1
            end
        end
    end
    data_all[b]=ana_data
end
length_range = (100:100:2000)
ana_data_all = zeros(length(b_range),length(length_range))
data =
for (bn,b) in enumerate(b_range)
    data=data_all[b]
    ana_data = zeros(length(data),length(length_range))
    for (i,d) in enumerate(data)
        for (j,len) in enumerate(length_range)
            p = get_param(d,len)
            if p > 0
                ana_data[i,j] = 1
            end
        end
    end
    a = (sum(ana_data,dims=1))
    a = [a[1,n] for n in range(1,length(length_range))]
    ana_data_all[bn,:] = a
    print(b)
end

for (bn, b) in enumerate(b_range)
    if bn==1
        plot(sum(data_all[b],dims=1)./2)
    else
        plot!(sum(data_all[b],dims=1)./2)
    end
    print(bn)
end
b=2.06
a = sum(data_all[b],dims=1)./50
a = [a[1,n] for n in range(1,length(length_range))]
plot!(length_range./6,a,linewidth=4,label="")
plot(sum(data_all[1.96],dims=1)./2)
length_range
xlabel!(L"\# Oscillations")
ylabel!(L"<T>")
savefig("C:/Users/emils/Dropbox/PHD/Speciale_artikel/nosc.pdf")




for (i,d) in enumerate(data)
    for (j,len) in enumerate(length_range)
        rs = Funcs.time_emb_r(vec(d[1,1:len*100]), 1)
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
        if f.param[2] > 0
            ana_data[i,j] = 1
        end
    end
    print(i)
end
