export async function loadTxt(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch in loadTxt. status: ${response.status}`);
  }
  console.log("hi", response);
}

export async function loadJSON(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch in loadJSON. status: ${response.status}`);
  }
  return await response.json();
}

export function convertRgbToHex(r, g, b) {
  r = Math.round(r * 255);
  g = Math.round(g * 255);
  b = Math.round(b * 255);

  const toHex = (n) => {
    const hex = n.toString(16);
    return hex.length === 1 ? "0" + hex : hex;
  };

  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}
