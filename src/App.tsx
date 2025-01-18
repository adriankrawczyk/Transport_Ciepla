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

import "./App.css";

const App: React.FC = () => {
  const [elements, setElements] = useState<number>(10);
  // Helper function to define the k(x) coefficient
  const k = (x: number): number => {
    if (x >= 0 && x <= 1) return 1; // k(x) = 1 for 0 <= x <= 1
    if (x > 1 && x <= 2) return 2; // k(x) = 2 for 1 < x <= 2
    throw new Error("Invalid k function argument!"); // Error for out-of-bound values
  };

  // Helper function to define the basis function e(i, x, h)
  const e = (i: number, x: number, h: number): number => {
    if (x < h * (i - 1) || x > h * (i + 1)) return 0; // e(i, x) is 0 outside the support
    if (x < h * i) return x / h - i + 1; // e(i, x) for x < h * i
    return -x / h + i + 1; // e(i, x) for x >= h * i
  };

  // Helper function for the derivative of the basis function e'(i, x, h)
  const ePrim = (i: number, x: number, h: number): number => {
    if (x < h * (i - 1) || x > h * (i + 1)) return 0; // e'(i, x) is 0 outside the support
    if (x < h * i) return 1 / h; // e'(i, x) for x < h * i
    return -1 / h; // e'(i, x) for x >= h * i
  };

  // Helper function to perform numerical integration using Gaussian quadrature
  const integrate = (
    f: (x: number) => number, // Function to integrate
    a: number, // Lower bound of integration
    b: number // Upper bound of integration
  ): number => {
    // Predefined weights and nodes for 10-point Gaussian quadrature
    const weights = [
      0.2955242247147529, 0.2955242247147529, 0.2692667193099963,
      0.2692667193099963, 0.219086362515982, 0.219086362515982,
      0.1494513491505806, 0.1494513491505806, 0.0666713443086881,
      0.0666713443086881,
    ];
    const nodes = [
      -0.1488743389816312, 0.1488743389816312, -0.4333953941292472,
      0.4333953941292472, -0.6794095682990244, 0.6794095682990244,
      -0.8650633666889845, 0.8650633666889845, -0.9739065285171717,
      0.9739065285171717,
    ];

    const mid = (a + b) / 2; // Midpoint of the interval
    const scale = (b - a) / 2; // Scaling factor for interval transformation

    // Sum over weighted function evaluations at transformed nodes
    return (
      scale *
      weights.reduce(
        (sum, weight, i) => sum + weight * f(mid + scale * nodes[i]),
        0
      )
    );
  };

  // Helper function to compute the stiffness matrix entries
  const B = (i: number, j: number, a: number, b: number, h: number): number => {
    // Integrand for the stiffness matrix entry
    const integrand = (x: number) => k(x) * ePrim(i, x, h) * ePrim(j, x, h);
    // Numerical integration and boundary term correction
    return integrate(integrand, a, b) - e(i, 0, h) * e(j, 0, h);
  };

  // Helper function to compute the load vector entries
  const L = (i: number, h: number): number => -20.0 * e(i, 0, h);

  // Main function to solve the system of equations
  const solve = () => {
    const h = 2.0 / elements; // Step size for subdivision
    const matrix: number[][] = Array(elements)
      .fill(0)
      .map(() => Array(elements).fill(0)); // Initialize stiffness matrix
    const vector: number[] = Array(elements).fill(0); // Initialize load vector

    // Assemble the stiffness matrix
    for (let i = 0; i < elements; i++) {
      for (let j = 0; j < elements; j++) {
        let a = 0.0,
          b = 0.0;

        // Determine integration bounds based on overlap of basis functions
        if (Math.abs(i - j) === 1) {
          a = 2.0 * Math.max(0.0, Math.min(i, j) / elements);
          b = 2.0 * Math.min(1.0, Math.max(i, j) / elements);
        } else if (i === j) {
          a = 2.0 * Math.max(0.0, (i - 1.0) / elements);
          b = 2.0 * Math.min(1.0, (i + 1.0) / elements);
        } else {
          continue; // Non-overlapping basis functions contribute 0
        }

        matrix[i][j] = B(i, j, a, b, h);
      }
    }

    // Assemble the load vector
    for (let i = 0; i < elements - 1; i++) {
      vector[i] = L(i, h);
    }
    vector[elements - 1] = 3.0; // Boundary condition at the last node

    // Solve the system using Gaussian elimination
    const solution = gaussianElimination(matrix, vector);
    solution.push(0); // Enforce boundary condition at x = 0

    // Return computed x and y values for visualization
    return {
      x: Array.from({ length: elements + 1 }, (_, i) => h * i),
      y: solution,
    };
  };

  // Function to perform Gaussian elimination
  const gaussianElimination = (A: number[][], b: number[]): number[] => {
    const n = A.length;
    const x: number[] = Array(n).fill(0);
    const augMatrix = A.map((row, i) => [...row, b[i]]); // Augmented matrix

    // Forward elimination
    for (let i = 0; i < n; i++) {
      let maxRow = i;
      for (let j = i + 1; j < n; j++) {
        if (Math.abs(augMatrix[j][i]) > Math.abs(augMatrix[maxRow][i])) {
          maxRow = j; // Pivot selection
        }
      }
      [augMatrix[i], augMatrix[maxRow]] = [augMatrix[maxRow], augMatrix[i]]; // Swap rows

      for (let j = i + 1; j < n; j++) {
        const factor = augMatrix[j][i] / augMatrix[i][i];
        for (let k = i; k <= n; k++) {
          augMatrix[j][k] -= factor * augMatrix[i][k];
        }
      }
    }

    // Back substitution
    for (let i = n - 1; i >= 0; i--) {
      let sum = augMatrix[i][n];
      for (let j = i + 1; j < n; j++) {
        sum -= augMatrix[i][j] * x[j];
      }
      x[i] = sum / augMatrix[i][i];
    }

    return x;
  };

  const solution = solve();

  const data = {
    labels: solution.x,
    datasets: [
      {
        label: "Temperature Distribution",
        data: solution.y,
        borderColor: "rgb(75, 192, 192)",
        tension: 0.1,
        pointRadius: 3,
        backgroundColor: "rgba(75, 192, 192, 0.2)",
      },
    ],
  };
  const handleElementsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;

    // Allow empty input for better typing experience
    if (value === "") {
      setElements(2);
      return;
    }

    const numValue = parseInt(value);

    // Validate the input
    if (!isNaN(numValue)) {
      // Clamp value between 2 and 50
      const clampedValue = Math.min(Math.max(numValue, 2), 50);
      setElements(clampedValue);
    }
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      title: {
        display: true,
        text: "Heat Transport Solution",
        font: {
          size: 20,
          weight: "bold" as const,
        },
        color: "#333",
      },
      legend: {
        position: "top" as const,
        labels: {
          font: {
            size: 14,
          },
        },
      },
    },
    scales: {
      y: {
        min: -30,
        max: 70,
        title: {
          display: true,
          text: "Temperature",
          font: {
            size: 14,
          },
        },
      },
      x: {
        title: {
          display: true,
          text: "Position",
          font: {
            size: 14,
          },
        },
      },
    },
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-b from-gray-100 to-gray-300 p-6">
      <div className="w-full max-w-4xl bg-white rounded-lg shadow-md p-8">
        <h1 className="text-3xl font-bold text-center mb-6 text-gray-800">
          Heat Transport Solver
        </h1>
        <div className="mb-6 flex justify-center items-center gap-4">
          <label
            htmlFor="elements"
            className="text-lg font-medium text-gray-700"
          >
            Number of elements:
          </label>
          <input
            id="elements"
            type="number"
            min="2"
            max="50"
            value={elements === 2 ? "" : elements}
            onChange={handleElementsChange}
            className="border rounded-lg px-3 py-2 w-24 text-gray-700 focus:ring focus:ring-blue-300 focus:outline-none"
          />
        </div>
        <div className="relative w-full" style={{ height: "50vh" }}>
          <Line className="h-full" data={data} options={options} />
        </div>
      </div>
    </div>
  );
};

export default App;
