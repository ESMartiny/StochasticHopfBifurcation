module Funcs
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

function sim_plus_fit(x_0, t_span, fun, f_σ, p, ndata, nbins)
    prob_sde_gluco = SDEProblem(fun, f_σ, x_0, t_span, p)
    sol = solve(prob_sde_gluco, SROCK2(); dt=0.05)
    f = fit_second_order(sol,p[2],ndata,nbins)
    return (f.param[2],  stderror(f)[2],confidence_interval(f, 0.05))
end

function sim_plus_fit_1degree(x_0, t_span, fun, f_σ, p, ndata, nbins)
    prob_sde_gluco = SDEProblem(fun, f_σ, x_0, t_span, p)
    sol = solve(prob_sde_gluco, SROCK2(); dt=0.05)
    f = fit_second_order_1var(sol,p[2],ndata,nbins)
    f = fit_second_order_1var(sol,p[2],ndata,nbins)
    return (f.param[1],  stderror(f)[1],confidence_interval(f, 0.05))
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
            #data_up[i,j] = f[3]
        end
        show(b," ")
    end
    return data,data_low,data_up
end
function raster_scan_no_fit(x_0, t_span, fun, f_σ, b_range, σ_range, ndata, nbins)
    data = Array{Any}(undef,(length(b_range),length(σ_range)))
    for (i, b) in enumerate(b_range)
        for (j, σ) in enumerate(σ_range)
            p = (10.0, b, σ)
            prob= SDEProblem(fun, f_σ, x_0, t_span, p)
            sol = solve(prob, SROCK2(); dt=0.005)
            rs = time_emb_r(vec(sol[1,:]), 10.0/5.0)
            rs_sorted = sort(rs)
            l = floor(Int64,length(rs_sorted)/1.1)
            h = fit(Histogram, rs_sorted[1:l],nbins=nbins)
            data[i,j] = h
        end
        show(b)
        show(" ")
    end
    return data
end
function raster_scan_no_fit_gluco(x_0, t_span, fun, f_σ, b_range, σ_range, ndata, nbins)
    data = Array{Any}(undef,(length(b_range),length(σ_range)))
    for (i, b) in enumerate(b_range)
        for (j, σ) in enumerate(σ_range)
            p = (0.1, b, σ)
            prob= SDEProblem(fun, f_σ, x_0, t_span, p)
            sol = solve(prob, SROCK2(); dt=0.05)
            rs = time_emb_r(vec(sol[1,:]), b)
            rs_sorted = sort(rs)
            l = floor(Int64,length(rs_sorted)/1.1)
            h = fit(Histogram, rs_sorted[1:l],nbins=nbins)
            data[i,j] = h
        end
        show(b)
        show(" ")
    end
    return data
end
function raster_scan_no_fit_normal(x_0, t_span, fun, f_σ, b_range, σ_range, ndata, nbins)
    data = Array{Any}(undef,(length(b_range),length(σ_range)))
    for (i, b) in enumerate(b_range)
        for (j, σ) in enumerate(σ_range)
            p = (1,1, b, 5, σ)
            prob= SDEProblem(fun, f_σ, x_0, t_span, p)
            sol = solve(prob, SROCK2(); dt=0.02)
            rs = time_emb_r(vec(sol[1,:]), 0)
            rs_sorted = sort(rs)
            l = floor(Int64,length(rs_sorted)/1.1)
            h = fit(Histogram, rs_sorted[1:l],nbins=nbins)
            data[i,j] = h
        end
        show(b)
        show(" ")
    end
    return data
end
function raster_scan_no_fit_brusselator(x_0, t_span, fun, f_σ, b_range, σ_range, ndata, nbins)
    data = Array{Any}(undef,(length(b_range),length(σ_range)))
    for (i, b) in enumerate(b_range)
        for (j, σ) in enumerate(σ_range)
            p = (1, b, σ)
            prob= SDEProblem(fun, f_σ, x_0, t_span, p)
            sol = solve(prob, SROCK2(); dt=0.05)
            rs = time_emb_r(vec(sol[1,:]), 1)
            rs_sorted = sort(rs)
            l = floor(Int64,length(rs_sorted)/1.1)
            h = fit(Histogram, rs_sorted[1:l],nbins=nbins)
            data[i,j] = h
        end
        show(b)
        show(" ")
    end
    return data
end

function raster_scan_1degree(x_0, t_span, fun, f_σ, b_range, σ_range, ndata, nbins)
    data = zeros(length(b_range),length(σ_range))
    data_low = zeros(length(b_range),length(σ_range))
    data_up = zeros(length(b_range),length(σ_range))
    for (i, b) in enumerate(b_range)
        for (j, σ) in enumerate(σ_range)
            p = (0.1, b, σ)
            f = sim_plus_fit_1degree(x_0, t_span, fun, f_σ, p, ndata, nbins)
            data[i,j] = f[1]
            data_low[i,j] = f[2]
            time_emb_r(time_series, f)
            #data_up[i,j] = f[3]
        end
        show(b)
    end
    return data,data_low,data_up
end
function raster_scan_par(x_0, t_span, fun, f_σ, b_range, σ_range, ndata, nbins)
    par_mat = zeros(length(b_range),length(σ_range))
    for (i, b) in enumerate(b_range)
        for (j, σ) in enumerate(σ_range)
            par_mat[i,j] = (0.1, b, σ)
        end
        show(b)
    end
    function sim_plus_fit_raster(p)
        f = sim_plus_fit(x_0, t_span, fun, f_σ, p, ndata, nbins)
    end

    return data,data_low,data_up
end

function raster_scan_4var(x_0, t_span, fun, f_σ, b_range, σ_range, ndata, nbins)
    data = zeros(length(b_range),length(σ_range))
    data_low = zeros(length(b_range),length(σ_range))
    data_up = zeros(length(b_range),length(σ_range))
    for (i, b) in enumerate(b_range)
        for (j, σ) in enumerate(σ_range)
            p = (1,1, b,10, σ)
            f = sim_plus_fit(x_0, t_span, fun, f_σ, p, ndata, nbins)
            data[i,j] = f[1]
            data_low[i,j] = f[2]
            data_up[i,j] = f[3]
        end
        show(b)
    end
    return data,data_low,data_up
end

function raster_of_sim(mat_of_hists, b_range, σ_range, ndata, nbins, type)
    other = Array{Any}(undef,(length(b_range),length(σ_range)))
    data = zeros(length(b_range),length(σ_range))
    if type=="kolmo"
        for (i, b) in enumerate(b_range)
            for (j, σ) in enumerate(σ_range)
                #?p = (0.1, b, σ)
                h = mat_of_hists[i,j]
                l = floor(Int64,length(h.weights)/ndata)
                empi = h.weights[1:l]
                max_val = empi[end]
                linear = [n * (max_val/l) for n in 1:l]
                empi = empi ./ sum(empi)
                linear = linear ./ sum(linear)
                ks_stat = 0
                for n in range(1, l)
                    ks_stat = max(ks_stat, abs(sum(empi[1:n]) - sum(linear[1:n])))
                end
                data[i,j] = ks_stat
            end
        end
        return data
    end
    if type=="second"
        for (i, b) in enumerate(b_range)
            for (j, σ) in enumerate(σ_range)
                #p = (0.1, b, σ)
                h = mat_of_hists[i,j]
                l = floor(Int64,length(h.weights)/ndata)
                empi = h.weights[1:l]
                @. model(x, p) = p[1] * x + p[2]*x^2
                a = (empi[end]/(length(empi)-1))
                xdata = 0:length(empi)-1
                p0 = [a,0]
                lb = [0, -Inf]
                ub = [length(empi), Inf]
                #f = curve_fit(model2, xdata, empi, p0,lower=lb, upper=ub)
                f = curve_fit(model, xdata, empi, p0)
                data[i,j] = f.param[2]
                other[i,j] = f
            end
        end
        return data, other
    end
    if type=="second_1degree"
        for (i, b) in enumerate(b_range)
            for (j, σ) in enumerate(σ_range)
                p = (0.1, b, σ)
                h = mat_of_hists[i,j]
                l = floor(Int64,length(h.weights)/ndata)
                empi = h.weights[1:l]
                max_val = empi[end]*1.0
                l = length(empi)-1
                ex = :(@. model(x, p) = p[1]*x^2 + ($max_val/$l - p[1]*$l)*x)
                eval(ex)
                xdata = 0:length(empi)-1
                p0 = [0.0]
                #f = curve_fit(model2, xdata, empi, p0,lower=lb, upper=ub)
                f = curve_fit(model, xdata, empi, p0)
                data[i,j] = f.param[1]
                other[i,j] = f
            end
        end
        return data,other
    end

end
function raster_of_sim_smart(mat_of_hists, b_range, σ_range, ndata, nbins, type)
    other = Array{Any}(undef,(length(b_range),length(σ_range)))
    data = zeros(length(b_range),length(σ_range))
    if type=="kolmo"
        for (i, b) in enumerate(b_range)
            for (j, σ) in enumerate(σ_range)
                #?p = (0.1, b, σ)
                h = mat_of_hists[i,j]
                toppoint = argmax(h.weights)
                l = floor(Int64,toppoint/ndata)
                empi = h.weights[1:l]
                max_val = empi[end]
                linear = [n * (max_val/l) for n in 1:l]
                empi = empi ./ sum(empi)
                linear = linear ./ sum(linear)
                ks_stat = 0
                for n in range(1, l)
                    ks_stat = max(ks_stat, abs(sum(empi[1:n]) - sum(linear[1:n])))
                end
                data[i,j] = ks_stat
            end
        end
        return data
    end
    if type=="second"
        for (i, b) in enumerate(b_range)
            for (j, σ) in enumerate(σ_range)
                #p = (0.1, b, σ)
                h = mat_of_hists[i,j]
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
                data[i,j] = f.param[2]
                other[i,j] = f
            end
        end
        return data, other
    end
    if type=="second_1degree"
        for (i, b) in enumerate(b_range)
            for (j, σ) in enumerate(σ_range)
                p = (0.1, b, σ)
                h = mat_of_hists[i,j]
                toppoint = argmax(h.weights)
                l = floor(Int64,toppoint/ndata)
                empi = h.weights[1:l]
                max_val = empi[end]*1.0
                l = length(empi)-1
                ex = :(@. model(x, p) = p[1]*x^2 + ($max_val/$l - p[1]*$l)*x)
                eval(ex)
                xdata = 0:length(empi)-1
                p0 = [0.0]
                #f = curve_fit(model2, xdata, empi, p0,lower=lb, upper=ub)
                f = curve_fit(model, xdata, empi, p0)
                data[i,j] = f.param[1]
                other[i,j] = f
            end
        end
        return data,other
    end

end
function kol_smirnov(sol, f,ndata, nbins)
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
function fit_second_order_1var(sol, fix, ndata, nbins)
    rs = time_emb_r(vec(sol[1,:]),fix)
    rs_sorted = sort(rs)
    n = floor(Int64,length(rs_sorted)/ndata)
    h = fit(Histogram, rs_sorted[1:n],nbins=nbins)
    empi = h.weights[1:end-3]
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

function fit_second_order_1var_plot(sol, f, ndata, nbins)
    rs = time_emb_r(vec(sol[1,:]),f)
    rs_sorted = sort(rs)
    n = floor(Int64,length(rs_sorted)/ndata)
    h = fit(Histogram, rs_sorted[1:n],nbins=nbins)
    empi = h.weights[1:end-3]
    max_val = empi[end]*1.0
    l = length(empi)-1
    ex = :(@. model(x, p) = p[1]*x^2 + ($max_val/$l - p[1]*$l)*x)
    eval(ex)
    xdata = 0:length(empi)-1
    p0 = [0.0]
    #f = curve_fit(model2, xdata, empi, p0,lower=lb, upper=ub)
    f = curve_fit(model, xdata, empi, p0)
    plot(empi)
    plot!(model(xdata, f.param))
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
function time_emb_poincare_filter(time_series)
    corr = StatsBase.autocor(time_series,range(1,stop=2000))
    τ = findfirst(corr.<0)
    x = time_series[1:end-τ]
    xτ = time_series[τ+1:end]
    side = (xτ .> x)
    crosses = side[1:(end-1)] .⊻ side[2:end]
    time_series = time_series[1:length(crosses)]
    crossvals = time_series[crosses]
    vals = []
    for i in range(2,length(crossvals))
        if abs(crossvals[i]-crossvals[i-1]) > 0.1
            push!(vals, crossvals[i])
        end
    end
    return vals
end
function time_emb_r(time_series, f)
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
function time_emb(time_series, f)
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
    return x, xτ
end

end
