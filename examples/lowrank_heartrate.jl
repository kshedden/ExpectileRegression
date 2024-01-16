using ExpectileRegression
using DataFrames
using CSV
using Dates
using Statistics
using Missings
using UnicodePlots

# Aggregate to this number of minutes
tres = 5
@assert 60 % tres == 0

# Number of distinct time units
ntime = div(24*60, tres)

hr = open("heartrate_seconds_merged.csv.gz") do io
    CSV.read(io, DataFrame)
end
hr = rename(hr, :Value=>:HR)
hr[:, :Date] = [DateTime(x, dateformat"m/d/y H:M:S p") for x in hr[:, :Time]]

# Round time to the nearest 'tres' minutes.
hr[:, :Datex] = round.(hr[:, :Date], Dates.Minute(tres))

# Create a variable called 'Day' that is only the year/month/day
hr[:, :Day] = round.(hr[:, :Datex], Dates.Day(1))

# Get the median HR in each 5-minute interval
hr = combine(groupby(hr, [:Id, :Datex]), :HR=>median, :Day=>first)
hr = rename(hr, :Day_first=>:Day, :HR_median=>:HR)
hr[!, :Day] = Date.(hr[:, :Day])

# Create an integer time index from 1 to 288.
ho = hour.(hr[:, :Datex])
mi = minute.(hr[:, :Datex])
hr[:, :tix] = div(60, tres) * ho + div.(mi, tres) .+ 1


function hr_pivot(hr)
    ha = (Id=Int[], Day=Date[])
    hb = []
    for hrx in groupby(hr, [:Id, :Day])
        id = first(hrx[:, :Id])
        day = first(hrx[:, :Day])
        row = missings(Float64, ntime)
        for r in eachrow(hrx)
            row[r.tix] = r.HR
        end
        push!(ha.Id, id)
        push!(ha.Day, day)
        push!(hb, row)
    end
    hb = hcat(hb...)
    ha = DataFrame(ha)
    return ha, copy(hb')
end

# Each row of HR contains heart rate data for one person on one day.
# HI contains meta-data about each person-day
HI, HR = hr_pivot(hr)

# Remove person/days with less than 50% observed data
rm = [mean(ismissing.(x)) for x in eachrow(HR)]
ii = findall(rm .<= 0.5)
HR = HR[ii, :]

HRx = copy(HR')

# Fit models for several expectiles
r = 1
md = []
for tau in [0.5, 0.9]
    m1 = fit(ExpectLR, HRx; r=r, tau=tau, verbosity=10)
    push!(md, m1)
end

for mm in md

    # Plot the additive diurnal structure
    plt = lineplot(mm.rcen)
    println(plt)

    # Plot the factor structure
    for j in 1:mm.r
        plt = lineplot(mm.V[:, j])
        println(plt)
    end
end
