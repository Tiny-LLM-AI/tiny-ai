export interface ArithmeticExample {
  /** For example "2+3=" */
  question: string;
  /** For example "5" */
  answer: string;
}

export interface ArithmeticDatasetFile {
  format: "arithmetic-dataset/v1";
  description: string;
  examples: ArithmeticExample[];
}

const QUESTION_PATTERN = /^(\d+)([+\-*/])(\d+)=$/u;

function calculate(left: number, operator: string, right: number): number {
  if (operator === "+") return left + right;
  if (operator === "-") return left - right;
  if (operator === "*") return left * right;
  return right === 0 ? Number.NaN : left / right;
}

/** Rejects malformed rows and rows whose answer is mathematically wrong. */
export function validateDataset(file: ArithmeticDatasetFile): ArithmeticExample[] {
  if (file.format !== "arithmetic-dataset/v1" || !Array.isArray(file.examples) || file.examples.length === 0) {
    throw new Error('Dataset must have format "arithmetic-dataset/v1" and a non-empty "examples" array.');
  }

  file.examples.forEach(({ question, answer }, index) => {
    const match = QUESTION_PATTERN.exec(question);
    if (!match || !/^-?\d+$/u.test(answer)) {
      throw new Error(`Example ${index} must look like { "question": "2+3=", "answer": "5" }.`);
    }
    const expected = calculate(Number(match[1]), match[2], Number(match[3]));
    if (String(expected) !== answer) {
      throw new Error(`Example ${index}: ${question} should be ${expected}, but the dataset says ${answer}.`);
    }
  });

  return file.examples;
}
