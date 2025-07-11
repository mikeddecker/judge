export function isNullOrWhiteSpace(input) {
  return typeof input !== 'string' || input.trim().length === 0;
}