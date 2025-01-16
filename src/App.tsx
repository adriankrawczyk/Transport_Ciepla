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
  b: number
): number => {
  // Two-point Gaussian quadrature weights and nodes
  const weights = [1, 1];
  const nodes = [-1 / Math.sqrt(3), 1 / Math.sqrt(3)];

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

// Finite Element Method implementation
const solveHeatTransport = (n: number): { x: number[]; u: number[] } => {
  const k = (x: number) => (x <= 1 ? 1 : 2); // Piecewise k(x)
  const length = 2;
  const h = length / n;

  // Nodes
  const nodes = Array.from({ length: n + 1 }, (_, i) => i * h);

  // Stiffness matrix and load vector
  const K = Array.from({ length: n + 1 }, () => Array(n + 1).fill(0));
  const F = Array(n + 1).fill(0);

  // Assemble stiffness matrix and load vector
  for (let i = 0; i < n; i++) {
    const x1 = nodes[i];
    const x2 = nodes[i + 1];

    const stiffnessLocal = [
      [1 / h, -1 / h],
      [-1 / h, 1 / h],
    ];

    for (let a = 0; a < 2; a++) {
      for (let b = 0; b < 2; b++) {
        const integrand = (x: number) => k(x) * stiffnessLocal[a][b];
        const integral = gaussianQuadrature(integrand, x1, x2);
        K[i + a][i + b] += integral;
      }
    }
  }

  // Apply boundary conditions
  K[n][n] = 1;
  F[n] = 3; // u(2) = 3

  F[0] += 20; // du(0)/dx + u(0) = 20

  // Solve the linear system
  const solveLinearSystem = (A: number[][], b: number[]): number[] => {
    const size = A.length;
    const x = Array(size).fill(0);

    for (let i = 0; i < size; i++) {
      let maxRow = i;
      for (let k = i + 1; k < size; k++) {
        if (Math.abs(A[k][i]) > Math.abs(A[maxRow][i])) {
          maxRow = k;
        }
      }

      [A[i], A[maxRow]] = [A[maxRow], A[i]];
      [b[i], b[maxRow]] = [b[maxRow], b[i]];

      for (let k = i + 1; k < size; k++) {
        const factor = A[k][i] / A[i][i];
        for (let j = i; j < size; j++) {
          A[k][j] -= factor * A[i][j];
        }
        b[k] -= factor * b[i];
      }
    }

    for (let i = size - 1; i >= 0; i--) {
      let sum = 0;
      for (let j = i + 1; j < size; j++) {
        sum += A[i][j] * x[j];
      }
      x[i] = (b[i] - sum) / A[i][i];
    }

    return x;
  };

  const u = solveLinearSystem(K, F);

  return { x: nodes, u };
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

  const options = {
    scales: {
      y: {
        min: -30,
        max: 70,
        ticks: {
          stepSize: 10,
        },
      },
    },
    maintainAspectRatio: true,
    responsive: true,
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
        <Line data={data} options={options} />
      </div>
    </div>
  );
};

export default App;
