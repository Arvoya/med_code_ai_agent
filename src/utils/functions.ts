
export function extractKeywords(text: string): string[] {
  // Remove common words and extract key medical terms
  const words = text.toLowerCase()
    .replace(/[^\w\s]/g, '')
    .split(/\s+/)
    .filter(word =>
      word.length > 3 &&
      !['what', 'which', 'when', 'where', 'this', 'that', 'with', 'from', 'have', 'code', 'represents'].includes(word)
    );

  return words;
}

export function calculateMatchScore(description: string, keywords: string[]): number {
  const descriptionLower = description.toLowerCase();
  let matchCount = 0;

  for (const keyword of keywords) {
    if (descriptionLower.includes(keyword)) {
      matchCount++;
    }
  }

  return keywords.length > 0 ? matchCount / keywords.length : 0;
}
