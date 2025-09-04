import { getColor } from "./utils"

export const transformDailyCounts = (dailyData, types, cummulative) => {
  let datapoints = Object.fromEntries(Object.keys(types).map(flt => [flt, []]))
  let indiviualOrCumulative = cummulative ? 'cumulative' : 'individual'

  let days = new Set()
  // Format values to an array per type
  Object.entries(dailyData).forEach(([day, typedDay]) => {
    Object.entries(typedDay[indiviualOrCumulative]).forEach(([flt, count]) => {
      datapoints[flt].push(count)
      days.add(day)
    })
  })

  let datasets = Object.entries(datapoints).map(([flt, counts]) => {
    return {
      label: types[flt],
      data: counts,
      borderColor: getColor(flt),
      fill: false,
      cubicInterpolationMode: 'monotone',
      tension: 0.4
    }
  })

  return {
    labels: Array.from(days),
    datasets: datasets
  };
}

export const getDailyChartOptions = (title, axisName = 'box count') => {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      title: {
        display: true,
        text: title
      },
    },
    interaction: {
      intersect: false,
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'day'
        }
      },
      y: {
        display: true,
        title: {
          display: true,
          text: axisName
        },
        // type: 'logarithmic',
        suggestedMin: 0,
        suggestedMax: 1,
        position: 'right'
      },
    }
  }
}

