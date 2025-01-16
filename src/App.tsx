import React, { useState } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

// Gaussian quadrature for numerical integration
const gaussianQuadrature = (
  f: (x: number) => number,
  a: number,
  b: number,
  points: number
): number => {
  // Generate weights and nodes dynamically for given number of points
  const weightsAndNodes = {
    2: {
      weights: [1, 1],
      nodes: [-1 / Math.sqrt(3), 1 / Math.sqrt(3)],
    },
  };

  const { weights, nodes } = weightsAndNodes[points];
  const midpoint = (a + b) / 2;
  const halfLength = (b - a) / 2;

  return (
    halfLength *
    nodes.reduce(
      (sum, node, i) => sum + weights[i] * f(midpoint + node * halfLength),
      0
    )
  );
};

// Function definitions
const piecewiseK = (x: number): number => (x <= 1 ? 1 : 2);
const basisFunction = (i: number, x: number, h: number): number => {
  if (x < h * (i - 1) || x > h * (i + 1)) return 0;
  if (x < h * i) return x / h - i + 1;
  return -(x / h) + i + 1;
};
const basisFunctionDerivative = (i: number, x: number, h: number): number => {
  if (x < h * (i - 1) || x > h * (i + 1)) return 0;
  if (x < h * i) return 1 / h;
  return -1 / h;
};

const assembleMatrix = (
  n: number,
  h: number,
  integrator: (
    f: (x: number) => number,
    a: number,
    b: number,
    points: number
  ) => number
): number[][] => {
  const matrix = Array.from({ length: n }, () => Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let a = Math.max(0, (i - 1) * h);
      let b = Math.min(2, (i + 1) * h);

      const integrand = (x: number) =>
        piecewiseK(x) *
        basisFunctionDerivative(i, x, h) *
        basisFunctionDerivative(j, x, h);

      matrix[i][j] = integrator(integrand, a, b, 2);
    }
  }
  return matrix;
};

const assembleVector = (
  n: number,
  h: number,
  integrator: (
    f: (x: number) => number,
    a: number,
    b: number,
    points: number
  ) => number
): number[] => {
  const vector = Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    vector[i] = -20 * basisFunction(i, 0, h);
  }
  vector[n - 1] = 3; // Dirichlet condition
  return vector;
};

const solveSystem = (matrix: number[][], vector: number[]): number[] => {
  const size = matrix.length;
  const x = Array(size).fill(0);

  for (let i = 0; i < size; i++) {
    let maxRow = i;
    for (let k = i + 1; k < size; k++) {
      if (Math.abs(matrix[k][i]) > Math.abs(matrix[maxRow][i])) {
        maxRow = k;
      }
    }

    [matrix[i], matrix[maxRow]] = [matrix[maxRow], matrix[i]];
    [vector[i], vector[maxRow]] = [vector[maxRow], vector[i]];

    for (let k = i + 1; k < size; k++) {
      const factor = matrix[k][i] / matrix[i][i];
      for (let j = i; j < size; j++) {
        matrix[k][j] -= factor * matrix[i][j];
      }
      vector[k] -= factor * vector[i];
    }
  }

  for (let i = size - 1; i >= 0; i--) {
    let sum = 0;
    for (let j = i + 1; j < size; j++) {
      sum += matrix[i][j] * x[j];
    }
    x[i] = (vector[i] - sum) / matrix[i][i];
  }

  return x;
};

const solveHeatTransport = (n: number): { x: number[]; u: number[] } => {
  const h = 2 / n;
  const nodes = Array.from({ length: n + 1 }, (_, i) => i * h);

  const matrix = assembleMatrix(n, h, gaussianQuadrature);
  const vector = assembleVector(n, h, gaussianQuadrature);

  const solution = solveSystem(matrix, vector);

  return { x: nodes, u: [...solution, 0] };
};

const App: React.FC = () => {
  const [n, setN] = useState(10);
  const { x, u } = solveHeatTransport(n);

  const data = {
    labels: x,
    datasets: [
      {
        label: "u(x)",
        data: u,
        borderColor: "rgba(75,192,192,1)",
        backgroundColor: "rgba(75,192,192,0.2)",
      },
    ],
  };

  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        height: "100vh",
        backgroundColor: "#f4f4f9",
      }}
    >
      <div
        style={{
          textAlign: "center",
          padding: "20px",
          background: "white",
          borderRadius: "8px",
          boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.1)",
        }}
      >
        <h1>Heat Transport Solver</h1>
        <div>
          <label htmlFor="n">Number of elements (n): </label>
          <input
            id="n"
            type="number"
            min="1"
            value={n}
            onChange={(e) => {
              const value = Math.max(1, Number(e.target.value));
              setN(value);
            }}
          />
        </div>
        <Line data={data} />
      </div>
    </div>
  );
};

export default App;
