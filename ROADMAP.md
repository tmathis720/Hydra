# HYDRA: Short-Term Roadmap for Incremental Test-Driven Development (TDD)

This roadmap outlines the short-term goals for the HYDRA project, focusing on test-driven development (TDD) principles. The goal is to incrementally add features while maintaining high test coverage, ensuring stability and modularity of the codebase.

## 1. Mesh Parsing and Verification

**Goal:** Ensure that the mesh parsing module is robust and tested for different input cases.

**Milestones:**

- **Task 1.1:** Refactor mesh loading to ensure flexibility and extensibility.
   - **Tests:** Mock different .msh2 inputs with various node and element counts, including edge cases (e.g., empty mesh, invalid formats).
**Test Implementation:**
    - Verify the correct number of nodes and elements are parsed.
    - Validate node and element properties (coordinates, IDs, neighbors).
- **Task 1.2:** Test for failure conditions.
    - **Tests:** Ensure error handling works (e.g., file not found, incorrect format).


## 2. Element Geometry Calculations

**Goal:** Isolate and verify geometry-related computations such as element areas and neighbor relationships.

**Milestones:**

- **Task 2.1:** Refactor area computations into a separate geometry_ops.rs module.
    - **Tests:** Use mock triangular element data to validate area calculations.
    - **Test Implementation:**
        - Test area computation for various types of triangles, including degenerate cases (zero area).
- **Task 2.2:** Test and validate neighbor relationships based on shared edges.
    - **Tests:** Use small mock meshes to validate that neighbors are correctly identified and assigned.


## 3. Flux Calculation Module

**Goal:** Implement and test flux calculations based on the computed element states and geometry.

**Milestones:**

- **Task 3.1:** Refactor flux calculation into flux_ops.rs under the transport_mod.
    - **Tests:** Use mock elements with pre-set states to validate that flux is correctly computed.
    - **Test Implementation:**
        - Test with various configurations of element states and areas.
        - Validate that positive and negative fluxes behave correctly based on physical intuition (e.g., higher state element should have an outgoing flux to a lower state element).
- **Task 3.2:** Test boundary conditions.
    - **Tests:** Simulate elements on the boundary with and without neighbors to ensure the solver handles boundaries correctly.


## 4. Linear Solver Development

**Goal:** Implement a robust linear solver module that operates on the computed fluxes and updates element states.

### Milestones:

- **Task 4.1:** Implement a LinearSolver struct that computes element fluxes and updates the state accordingly.
    - **Tests:** Verify that fluxes computed in the previous step are applied to element states.
	- **Test Implementation:**
		- Use small mock meshes with known states to validate that the state is correctly updated after each solver step.
		- Validate that the flux application behaves consistently across multiple time steps.


## 5. Explicit Time-Stepping Module

**Goal:** Develop a flexible time-stepping framework, starting with an explicit Euler scheme.

### Milestones:

- **Task 5.1:** Implement ExplicitEuler time-stepping in time_stepping_mod.
	- **Tests:** Ensure that the time stepper advances the solution by applying the solver and updating element states.
**Test Implementation:**
	- Validate small time-step increments on mock meshes (e.g., diffusion across elements).
	- Test the behavior over multiple steps, ensuring consistency in state updates.
- **Task 5.2:** Extend test coverage for time-stepping.
	- **Tests:** Verify behavior for varying time step sizes, testing both stability and accuracy.


## 6. Integration of Tests Across Modules

**Goal:** Combine individual module tests into integration tests to verify end-to-end functionality.

### Milestones:

- **Task 6.1:** Design integration tests that simulate the entire pipeline from mesh parsing to flux computation, solving, and time-stepping.
	- **Tests:** Run integration tests on small mock meshes to ensure all components work together.
	- **Test Implementation:**
        - Verify the complete simulation lifecycle, ensuring that the solver behaves as expected over multiple time steps.
        - Validate the overall conservation of flux or other relevant physical properties.


## 7. Error Handling and Edge Case Testing

**Goal:** Identify and test edge cases, focusing on failure modes and robustness.

### Milestones:

- **Task 7.1:** Test invalid inputs (e.g., degenerate meshes, zero-area elements, boundary edge cases).
	- **Tests:** Validate that the program fails gracefully, producing clear error messages.


## 8. Continuous Integration (CI) Integration

**Goal:** Set up automated testing to enforce TDD principles using a CI pipeline.

### Milestones:

- **Task 8.1:** Set up GitHub Actions or a similar CI tool.
	- **Tests:** Ensure all tests (unit and integration) are run automatically on every push or pull request.


## Summary

This roadmap focuses on incremental development with continuous testing, keeping the code modular and well-tested. Each component is isolated and validated before integrating it into the broader framework, maintaining alignment with test-driven development principles.