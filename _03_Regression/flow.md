# Gradient Descent and Cost Function – Simple Notes

## 1. Overview (Flow of the Process)

We start with a dataset plotted on a graph. Our goal is to find the **best-fit line** that represents the data.

### Step-by-step flow:

1. We choose some initial values for:
   - Theta 0 (intercept)
   - Theta 1 (slope)

2. Using these values, we draw a line on the graph.

3. We check how far this line is from actual data points.
   - This difference is called **error**.

4. We calculate the total error for all points.
   - This is called the **cost function value**.

5. We repeat this process with different values of Theta 0 and Theta 1.
   - Each pair gives a different line and different cost.

6. Our goal:
   - Find the **lowest cost value**
   - Because lower cost means better fit

7. Now we use **Gradient Descent**:
   - It helps us automatically adjust Theta values
   - It moves step by step toward minimum cost

8. Finally:
   - We get the best Theta values
   - That gives us the **best-fit line**

---

## 2. Important Definitions

### Dataset
A collection of points (x, y) plotted on a graph.

**Example:**
House size vs house price

---

### Best-Fit Line
A line that passes as close as possible to all data points.

**Example:**
A line that shows the trend of increasing house prices with size.

---

### Theta 0 (Intercept)
The starting point of the line on the Y-axis.

**Simple meaning:**
Where the line crosses the vertical axis.

---

### Theta 1 (Slope)
Shows how steep the line is.

**Simple meaning:**
How much Y changes when X increases.

---

### Error (Residual Error)
The difference between:
- Actual value (real data)
- Predicted value (from line)

**Example:**
If actual price = 100  
Predicted price = 90  
Error = 10

---

### Cost Function
It tells us how good or bad our line is.

- High cost → Bad fit  
- Low cost → Good fit  

**Goal:**
Minimize this value

---

### Gradient Descent
An algorithm that helps us find the best values of Theta 0 and Theta 1.

**How it works:**
- Starts with random values
- Slowly improves them step by step
- Moves toward lowest cost

**Simple idea:**
Like going downhill to reach the lowest point in a valley

---

## 3. Important Clarification

### Do we choose Theta values ourselves?

Yes, but only at the start.

- We give **random initial values**
- After that, Gradient Descent updates them automatically

So:
- Human → gives starting point  
- Algorithm → finds best values  

---

## 4. Final Understanding

- Different Theta values → Different lines  
- Different lines → Different errors  
- Errors → Cost function value  
- Gradient Descent → Finds minimum cost  
- Minimum cost → Best-fit line  

---

## 5. Real-Life Example

Imagine throwing a line on a graph randomly.

- First try: Line is far from points → high error  
- Second try: Better line → lower error  
- After many tries: Best line → lowest error  

Gradient Descent does this automatically for you.

---

## 6. Key Points to Remember

- We do NOT manually test all lines
- Gradient Descent finds best solution efficiently
- Lower cost = better model
- Learning is iterative (step-by-step improvement)
