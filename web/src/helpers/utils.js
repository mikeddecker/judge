export function isNullOrWhiteSpace(input) {
  return typeof input !== 'string' || input.trim().length === 0;
}

function getRandomInt(max) {
  return Math.floor(Math.random() * max);
}

export function round2decimals(i) {
  return Math.round(i * 100) / 100
}

export function formatPercentage(value) {
  return (value * 100).toFixed(1) + '%';
}

export function getColor(skillprop) {
  switch (skillprop) {
    case 1:
    case '1':
      return 'rgb(123, 222, 123)'
    case 2:
    case '2':
      return 'rgb(123, 222, 222)'
    case 'Total':
      return `rgb(150, 50, 0)`
    case 'Skill':
      return `rgb(0, 20, 20)`
    default:
      let greencolor = 80 + getRandomInt(175)
      return `rgb(${greencolor * Math.random()}, ${190 + getRandomInt(65)}, ${greencolor})`
      return `rgb(${getRandomInt(255)}, ${getRandomInt(255)}, ${getRandomInt(255)})`
  }
}

export async function sleep(ms) { await new Promise(r => setTimeout(r, ms)); }
