'use strict';

exports.arrayAdd = arrayAdd;
exports.arraySubtract = arraySubtract;
exports.arrayMultiply = arrayMultiply;
exports.sigmoid = sigmoid;
exports.sigmoidGradient = sigmoidGradient;

function arrayAdd(array1, array2) {
  const l1 = array1.length;
  const l2 = array2.length;
  if (l1 !== l2) {
    throw Error('Can\'t add arrays of length '+l1 +' and '+l2);
  }

  const result = [];
  for (let i = 0; i < l1; i++) {
    result.push(array1[i] + array2[i]);
  }
  return result;
}

function arraySubtract(array1, array2) {
  const l1 = array1.length;
  const l2 = array2.length;
  if (l1 !== l2) {
    throw Error('Can\'t subtract arrays of length '+l1 +' and '+l2);
  }

  const result = [];
  for (let i = 0; i < l1; i++) {
    result.push(array1[i] - array2[i]);
  }
  return result;
}

function arrayMultiply(array1, array2) {
  const l1 = array1.length;
  const l2 = array2.length;
  if (l1 !== l2) {
    throw Error('Can\'t multiply arrays of length '+l1 +' and '+l2);
  }

  let result = 0;
  for (let i = 0; i < l1; i++) {
    result += array1[i] * array2[i];
  }
  return result;
}

function sigmoid(z) {
  return 1 / (1 + Math.exp(z * -1));
}

function sigmoidGradient(z) {
  return sigmoid(z) * (1 - sigmoid(z));
}
