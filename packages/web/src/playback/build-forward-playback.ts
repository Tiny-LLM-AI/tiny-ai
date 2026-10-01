import { argMax, getRow, matrixFromRows, sliceColumns, type Matrix, type Model, type PredictionStep } from "@math-llm/core";
import { displayToken, formatPercent, indexLabels } from "../utils/format";
import { contextRowLabels, vocabularyLabels } from "../utils/labels";
import {
  addFrames,
  layerNormFrames,
  matMulFrames,
  num,
  panel,
  rowCells,
  rowMatrix,
  RevealTracker,
  softmaxFrames,
  type Operand,
} from "./frame-builders";
import type { Frame, PlaybackStage } from "./frame-types";

const lastRowOnly = (matrix: Matrix): Matrix => matrixFromRows([getRow(matrix, matrix.rows - 1)]);

/** Turns one forward pass into an animation: every stage of the network, frame by frame. */
export function buildForwardPlayback(model: Model, step: PredictionStep): PlaybackStage[] {
  const { config, parameters, tokenizer } = model;
  const { trace } = step;
  const vocabulary = vocabularyLabels(model);
  const positions = contextRowLabels(model, trace.tokenIds);
  const lastRow = config.contextLength - 1;
  const lastLabel = positions[lastRow];
  const features = indexLabels(config.embeddingSize);
  const headSize = config.embeddingSize / config.headCount;
  const stages: PlaybackStage[] = [];

  const op = (key: string, title: string, matrix: Matrix, rowLabels: string[], columnLabels: string[], extra: Partial<Operand> = {}): Operand => ({
    key,
    title,
    matrix,
    rowLabels,
    columnLabels,
    ...extra,
  });
  const activation = (key: string, title: string, matrix: Matrix, columnLabels = indexLabels(matrix.cols)) =>
    op(key, title, matrix, positions, columnLabels);
  const weight = (key: string, title: string, matrix: Matrix) =>
    op(key, title, matrix, indexLabels(matrix.rows), indexLabels(matrix.cols));
  const bias = (key: string, title: string, matrix: Matrix) => op(key, title, matrix, ["bias"], indexLabels(matrix.cols));

  // 1. Tokenize
  {
    const idMatrix = matrixFromRows([trace.tokenIds]);
    const ids = op("ids", "token ids", idMatrix, ["id"], positions, { colorMode: "plain", digits: 0 });
    const vocabularyIds = op("vocab", "vocabulary (character → id)", matrixFromRows([vocabulary.map((_, id) => id)]), ["id"], vocabulary, {
      colorMode: "plain",
      digits: 0,
    });
    const revealed = new RevealTracker(config.contextLength);
    const frames: Frame[] = [
      {
        narration: `The input is the text "${step.contextText}". A neural network can only work with numbers, so each character is replaced by its position in the vocabulary. The context always has ${config.contextLength} slots; empty slots on the left are filled with <PAD> (id 0).`,
        calculation: [],
        panels: [panel(vocabularyIds), panel(ids, { isCellRevealed: revealed.snapshot() })],
      },
    ];
    trace.tokenIds.forEach((id, position) => {
      revealed.add(0, position);
      frames.push({
        narration: `Slot ${position} holds "${displayToken(tokenizer.vocabulary[id])}". Look it up in the vocabulary.`,
        calculation: [`"${displayToken(tokenizer.vocabulary[id])}" → id ${id}`],
        panels: [
          panel(vocabularyIds, { highlightedCells: [{ row: 0, col: id }] }),
          panel(ids, { isCellRevealed: revealed.snapshot(), activeCells: [{ row: 0, col: position }] }),
        ],
      });
    });
    stages.push({ title: "Text → token ids", shortTitle: "Tokenize", formula: "character → id", frames });
  }

  // 2. Token embedding lookup
  {
    const table = op("tokenEmbedding", "tokenEmbedding (learned table)", parameters.tokenEmbedding, vocabulary, features);
    const vectors = activation("tokenVectors", "token vectors", trace.tokenVectors, features);
    const revealed = new RevealTracker(config.embeddingSize);
    const frames: Frame[] = [
      {
        narration: `An id is just a label. To give the model something to compute with, each id picks one row of a learned table: ${config.embeddingSize} numbers per character. No arithmetic here, only copying a row.`,
        calculation: [],
        panels: [panel(table), panel(vectors, { isCellRevealed: revealed.snapshot() })],
      },
    ];
    trace.tokenIds.forEach((id, position) => {
      revealed.addRow(position);
      frames.push({
        narration: `Slot ${position} has id ${id} ("${vocabulary[id]}"), so copy row ${vocabulary[id]} of the table.`,
        calculation: [`token vectors[${positions[position]}] = tokenEmbedding[row ${id}] = [${getRow(parameters.tokenEmbedding, id).slice(0, 4).map(num).join(", ")}, …]`],
        panels: [
          panel(table, { highlightedRows: [id] }),
          panel(vectors, { isCellRevealed: revealed.snapshot(), activeCells: rowCells(position, config.embeddingSize) }),
        ],
      });
    });
    stages.push({ title: "Look up token vectors", shortTitle: "Embed", formula: "vector = tokenEmbedding[id]", frames });
  }

  // 3. Add position vectors
  stages.push({
    title: "Add position vectors",
    shortTitle: "+ Position",
    formula: "x = token vector + position vector",
    frames: addFrames({
      intro: `"3" in slot 0 and "3" in slot 2 would otherwise look identical. Each slot has its own learned position vector, which is added to the token vector.`,
      a: activation("tokenVectors", "token vectors", trace.tokenVectors, features),
      aName: "token",
      b: op("positionEmbedding", "positionEmbedding (learned)", trace.positionVectors, positions.map((_, i) => `pos ${i}`), features),
      bName: "position",
      result: activation("x", "x = block input", trace.embeddingSum, features),
      resultName: "x",
      rows: trace.tokenIds.map((_, i) => i),
    }),
  });

  trace.blocks.forEach((blockTrace, blockIndex) => {
    const block = parameters.blocks[blockIndex];
    const prefix = trace.blocks.length > 1 ? `Block ${blockIndex + 1}: ` : "";
    const x = activation("x", "x (block input)", blockTrace.input, features);

    // 4. Q, K, V
    const projection = (key: "query" | "key" | "value", letter: string, detail: "full" | "summary", meaning: string) =>
      matMulFrames({
        intro: `${letter} = x · W${letter} + b${letter}. ${meaning} Each of the ${config.contextLength} rows of x is multiplied with the weight matrix W${letter} [${config.embeddingSize} × ${config.embeddingSize}].`,
        left: x,
        leftName: "x",
        right: weight(`W${letter}`, `W${letter} (learned)`, block[`${key}Weight`]),
        rightName: `W${letter}`,
        bias: bias(`b${letter}`, `b${letter}`, block[`${key}Bias`]),
        biasName: `b${letter}`,
        result: activation(letter, letter, blockTrace[key], features),
        resultName: letter,
        focusRow: lastRow,
        detail,
        otherRowsFirst: true,
        otherRowsNote:
          detail === "summary"
            ? `${letter} is calculated exactly like Q, only with different learned weights. All ${config.contextLength * config.embeddingSize} cells at once.`
            : "Same calculation; only the last row is shown in detail.",
      });
    stages.push({
      title: `${prefix}Query  Q = x · WQ + bQ`,
      shortTitle: "Q",
      formula: "Q[i, j] = Σₖ x[i, k] × WQ[k, j] + bQ[j]",
      frames: projection("query", "Q", "full", 'Q ("query") is what each position is looking for.'),
    });
    stages.push({
      title: `${prefix}Key and Value  K = x · WK + bK,  V = x · WV + bV`,
      shortTitle: "K, V",
      formula: "same recipe as Q, different weights",
      frames: [
        ...projection("key", "K", "summary", 'K ("key") is what each position offers to others.'),
        ...projection("value", "V", "summary", 'V ("value") is the information each position hands over.'),
      ],
    });

    blockTrace.heads.forEach((head, headIndex) => {
      const columns = indexLabels(headSize).map((label) => `${headIndex * headSize + Number(label)}`);
      const detailed = headIndex === 0;
      const headName = `head ${headIndex + 1}`;
      const q = activation(`q${headIndex}`, `Q (${headName} columns)`, head.query, columns);
      const k = activation(`k${headIndex}`, `K (${headName} columns)`, head.key, columns);
      const v = activation(`v${headIndex}`, `V (${headName} columns)`, head.value, columns);
      const scale = 1 / Math.sqrt(headSize);

      // 5. Attention scores
      const scores = matMulFrames({
        intro: `${headName} uses columns ${columns[0]}–${columns[columns.length - 1]} of Q, K and V. The score of position i towards position j is the dot product of row i of Q and row j of K: a large score means "j has what i is looking for". It is divided by √${headSize} to keep numbers small.`,
        left: q,
        leftName: "Q",
        right: k,
        rightName: "K",
        rightTransposed: true,
        scale: { factor: scale, description: `1/√${headSize}` },
        result: activation(`scores${headIndex}`, "scores", head.scores, positions),
        resultName: "score",
        focusRow: lastRow,
        detail: detailed ? "full" : "summary",
        otherRowsFirst: false,
        otherRowsNote: detailed
          ? `The other rows are calculated the same way. Only the last row matters for the prediction, because the model predicts from the last position (${lastLabel}).`
          : `${headName} does exactly what head 1 did, with its own columns of Q and K.`,
      });

      // 6. Mask + softmax
      const masked = activation(`masked${headIndex}`, "masked scores", head.maskedScores, positions);
      const weights = op(`weights${headIndex}`, "attention weights", head.attentionWeights, positions, positions, {
        colorMode: "probability",
      });
      const maskFrame: Frame = {
        narration: "Causal mask: a position may not look at positions to its right (the future). Those scores are replaced by −∞.",
        calculation: ["score[i, j] = −∞   whenever j > i"],
        panels: [panel(activation(`scores${headIndex}`, "scores", head.scores, positions)), panel(masked)],
      };
      const softmax = softmaxFrames({
        intro: `Softmax turns the last row of scores (${lastLabel} looking at every position) into percentages that add up to 100%.`,
        labels: positions,
        scores: getRow(head.maskedScores, lastRow),
        scoresTitle: `masked scores, row ${lastLabel}`,
        resultTitle: "attention weights",
        closing: "This row now says how much attention the last position pays to every position.",
      });
      const allWeights: Frame = {
        narration: "Every row gets the same softmax treatment. Each row adds up to 100%.",
        calculation: [],
        panels: [panel(masked), panel(weights, { highlightedRows: [lastRow] })],
      };
      stages.push({
        title: `${prefix}${headName}: attention scores`,
        shortTitle: `H${headIndex + 1} scores`,
        formula: `score = Q · Kᵀ / √${headSize}`,
        frames: scores,
      });
      stages.push({
        title: `${prefix}${headName}: mask + softmax`,
        shortTitle: `H${headIndex + 1} softmax`,
        formula: "weight = e^score / Σ e^score",
        frames: detailed ? [maskFrame, ...softmax, allWeights] : [maskFrame, allWeights],
      });

      // 7. Weighted sum of values
      stages.push({
        title: `${prefix}${headName}: weighted sum of values`,
        shortTitle: `H${headIndex + 1} · V`,
        formula: "output = weights · V",
        frames: matMulFrames({
          intro: "Each position collects information: a weighted average of all V rows, using the attention weights as percentages.",
          left: weights,
          leftName: "weight",
          right: v,
          rightName: "V",
          result: activation(`headOut${headIndex}`, `${headName} output`, head.output, columns),
          resultName: "out",
          focusRow: lastRow,
          detail: detailed ? "full" : "summary",
          otherRowsFirst: false,
          otherRowsNote: "The other rows are calculated the same way.",
        }),
      });
    });

    // 8. Concatenate + output projection
    const concat = activation("concat", "heads side by side", blockTrace.concatenatedHeads, features);
    stages.push({
      title: `${prefix}Combine heads`,
      shortTitle: "Combine",
      formula: "attn = concat(heads) · WO + bO",
      frames: [
        {
          narration: `The ${config.headCount} head outputs are placed side by side: ${config.headCount} × ${headSize} = ${config.embeddingSize} columns. No arithmetic, just gluing.`,
          calculation: [],
          panels: blockTrace.heads.map((head, index) =>
            panel(activation(`h${index}`, `head ${index + 1} output`, head.output)),
          ).concat(panel(concat)),
        },
        ...matMulFrames({
          intro: "A learned matrix WO mixes the information that the heads found.",
          left: concat,
          leftName: "concat",
          right: weight("WO", "WO (learned)", block.attentionOutputWeight),
          rightName: "WO",
          bias: bias("bO", "bO", block.attentionOutputBias),
          biasName: "bO",
          result: activation("attn", "attention output", blockTrace.attentionOutput, features),
          resultName: "attn",
          focusRow: lastRow,
          detail: "summary",
          otherRowsFirst: false,
          otherRowsNote: "Same row × column recipe as Q, for all cells.",
        }),
      ],
    });

    // 9. Residual + layer norm
    stages.push({
      title: `${prefix}Residual + layer norm`,
      shortTitle: "Add & norm 1",
      formula: "h = LayerNorm(x + attn)",
      frames: [
        ...addFrames({
          intro: "Residual connection: add the attention result to the original input, so nothing from the input gets lost.",
          a: x,
          aName: "x",
          b: activation("attn", "attention output", blockTrace.attentionOutput, features),
          bName: "attn",
          result: activation("res1", "x + attn", blockTrace.attentionResidual, features),
          resultName: "sum",
          rows: [lastRow],
          otherRowsNote: "All other rows are added the same way.",
        }),
        ...layerNormFrames({
          intro: `Layer norm rescales each row so its numbers are centred around 0 with a spread of about 1. Shown for row ${lastLabel}.`,
          rowLabel: lastLabel,
          input: getRow(blockTrace.attentionResidual, lastRow),
          normalized: getRow(blockTrace.attentionNorm.normalized, lastRow),
          output: getRow(blockTrace.attentionNorm.output, lastRow),
          scale: getRow(block.attentionNormScale, 0),
          shift: getRow(block.attentionNormShift, 0),
          standardDeviation: blockTrace.attentionNorm.standardDeviations[lastRow],
        }),
      ],
    });

    // 10. Feed-forward
    const h = activation("h", "h (after norm)", blockTrace.attentionNorm.output, features);
    const hidden = indexLabels(config.feedForwardSize);
    const reluFrame = (): Frame => {
      const before = lastRowOnly(blockTrace.feedForwardHidden);
      const after = lastRowOnly(blockTrace.feedForwardActivated);
      const values = getRow(before, 0);
      const negatives = values.map((value, col) => ({ value, col })).filter((entry) => entry.value < 0);
      return {
        narration: `ReLU: every negative number becomes 0, positive numbers stay. This simple bend is what lets the network learn non-linear rules. ${negatives.length} of ${values.length} values in row ${lastLabel} are switched off.`,
        calculation: [
          "ReLU(v) = max(0, v)",
          ...values.slice(0, 4).map((value, col) => `ReLU(${num(value)}) = ${num(Math.max(0, value))}   (unit ${col})`),
          "…",
        ],
        panels: [
          panel(op("pre", `before ReLU, row ${lastLabel}`, before, [lastLabel], hidden), {
            highlightedCells: negatives.map((entry) => ({ row: 0, col: entry.col })),
          }),
          panel(op("post", `after ReLU, row ${lastLabel}`, after, [lastLabel], hidden), { activeCells: rowCells(0, values.length) }),
        ],
      };
    };
    stages.push({
      title: `${prefix}Feed-forward, part 1`,
      shortTitle: "FF 1 + ReLU",
      formula: "u = ReLU(h · W1 + b1)",
      frames: [
        ...matMulFrames({
          intro: `Each position is now processed on its own by a small 2-layer network. Layer 1 expands ${config.embeddingSize} numbers to ${config.feedForwardSize}.`,
          left: h,
          leftName: "h",
          right: weight("W1", "W1 (learned)", block.feedForwardInputWeight),
          rightName: "W1",
          bias: bias("b1", "b1", block.feedForwardInputBias),
          biasName: "b1",
          result: activation("u", "h · W1 + b1", blockTrace.feedForwardHidden, hidden),
          resultName: "u",
          focusRow: lastRow,
          detail: "full",
          otherRowsFirst: false,
          otherRowsNote: "The other rows are calculated the same way.",
        }),
        reluFrame(),
      ],
    });
    stages.push({
      title: `${prefix}Feed-forward, part 2`,
      shortTitle: "FF 2",
      formula: "f = u · W2 + b2",
      frames: matMulFrames({
        intro: `Layer 2 shrinks the ${config.feedForwardSize} numbers back to ${config.embeddingSize}.`,
        left: activation("u", "ReLU output", blockTrace.feedForwardActivated, hidden),
        leftName: "u",
        right: weight("W2", "W2 (learned)", block.feedForwardOutputWeight),
        rightName: "W2",
        bias: bias("b2", "b2", block.feedForwardOutputBias),
        biasName: "b2",
        result: activation("f", "feed-forward output", blockTrace.feedForwardOutput, features),
        resultName: "f",
        focusRow: lastRow,
        detail: "summary",
        otherRowsFirst: false,
        otherRowsNote: "Same row × column recipe, all cells at once.",
      }),
    });
    stages.push({
      title: `${prefix}Residual + layer norm`,
      shortTitle: "Add & norm 2",
      formula: "out = LayerNorm(h + f)",
      frames: [
        ...addFrames({
          intro: "Second residual connection: add the feed-forward result to its input.",
          a: h,
          aName: "h",
          b: activation("f", "feed-forward output", blockTrace.feedForwardOutput, features),
          bName: "f",
          result: activation("res2", "h + f", blockTrace.feedForwardResidual, features),
          resultName: "sum",
          rows: [lastRow],
          otherRowsNote: "All other rows are added the same way.",
        }),
        ...layerNormFrames({
          intro: `Second layer norm, row ${lastLabel}. The result is the output of the block.`,
          rowLabel: lastLabel,
          input: getRow(blockTrace.feedForwardResidual, lastRow),
          normalized: getRow(blockTrace.feedForwardNorm.normalized, lastRow),
          output: getRow(blockTrace.feedForwardNorm.output, lastRow),
          scale: getRow(block.feedForwardNormScale, 0),
          shift: getRow(block.feedForwardNormShift, 0),
          standardDeviation: blockTrace.feedForwardNorm.standardDeviations[lastRow],
        }),
      ],
    });
  });

  // 11. Logits
  stages.push({
    title: "Scores for every character (logits)",
    shortTitle: "Logits",
    formula: "logit[c] = Σₖ last[k] × Wout[k, c] + bout[c]",
    frames: matMulFrames({
      intro: `Only the last row (${lastLabel}) is used to predict the next character. It is multiplied by Wout, which has one column per vocabulary character, giving ${vocabulary.length} scores.`,
      left: op("last", `last row (${lastLabel})`, trace.finalVector, [lastLabel], features),
      leftName: "last",
      right: op("Wout", "Wout (learned)", parameters.outputWeight, features, vocabulary),
      rightName: "Wout",
      bias: op("bout", "bout", parameters.outputBias, ["bias"], vocabulary),
      biasName: "bout",
      result: op("logits", "logits", trace.logits, ["score"], vocabulary),
      resultName: "logit",
      focusRow: 0,
      detail: "full",
      otherRowsFirst: false,
      otherRowsNote: "",
    }),
  });

  // 12. Probabilities + pick
  const probabilities = getRow(trace.probabilities, 0);
  const best = argMax(probabilities);
  stages.push({
    title: "Probabilities and the chosen character",
    shortTitle: "Pick",
    formula: "p = softmax(logits),  answer = argmax p",
    frames: [
      ...softmaxFrames({
        intro: "Softmax turns the scores into probabilities, one per character.",
        labels: vocabulary,
        scores: getRow(trace.logits, 0),
        scoresTitle: "logits",
        resultTitle: "probabilities",
        closing: "Every character now has a probability.",
      }),
      {
        narration: `Pick the character with the highest probability: "${vocabulary[best]}". It is appended to the text and the whole process starts again for the next character${step.chosenToken === "<END>" ? " (here <END> means the answer is finished)" : ""}.`,
        calculation: [`argmax → "${vocabulary[best]}" with ${formatPercent(probabilities[best])}`, `"${step.contextText}" + "${displayToken(step.chosenToken)}"`],
        panels: [
          panel(op("probs", "probabilities", rowMatrix(probabilities), [""], vocabulary, { colorMode: "probability", digits: 3 }), {
            activeCells: [{ row: 0, col: best }],
          }),
        ],
      },
    ],
  });

  return stages;
}
