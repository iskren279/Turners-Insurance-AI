function calculateCarValue(model, year) {
  if (
    typeof model !== "string" ||
    model.trim() === "" ||
    !Number.isInteger(year) ||
    year < 0
  ) {
    throw new Error("there is an error");
  }

  let alphabetValue = 0;

  for (const character of model) {
    if (/[a-zA-Z]/.test(character)) {
      alphabetValue += character.toLowerCase().charCodeAt(0) - 96;
    }
  }

  return alphabetValue * 100 + year;
}

export default calculateCarValue;
