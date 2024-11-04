# Detailed Report on Integrating Expanded Equations into the Hydra Program

## Introduction

The Hydra program is designed as a general framework for simulating fluid dynamics using the Finite Volume Method (FVM) on unstructured meshes. The `Equation` module plays a central role in formulating and solving the governing equations, such as momentum, continuity, energy conservation, and turbulence models.

This report provides a comprehensive overview of the `Equation` module's current structure and outlines a detailed plan for integrating additional components—specifically the energy equation and turbulence models—into the Hydra program. The goal is to extend the program's capabilities to support a broader range of physical phenomena while maintaining modularity and scalability.

---

## 1. Overview of the `Equation` Module

### Purpose

The `Equation` module encapsulates the mathematical representation of physical laws governing fluid flow. It is responsible for assembling and solving equations related to:

- **Momentum and Continuity**: Represented by the Navier-Stokes equations for fluid motion.
- **Energy Conservation**: Governing thermal effects in the fluid.
- **Turbulence Models**: Providing closure for the Reynolds-Averaged Navier-Stokes (RANS) equations.

### Key Responsibilities

- **Flux Computation**: Calculating fluxes at cell faces using appropriate numerical schemes.
- **Gradient Calculation**: Estimating gradients of scalar and vector fields within cells.
- **Equation Assembly**: Building the discrete equations that form the global system to be solved.
- **Boundary Condition Application**: Incorporating boundary conditions into the equations.

---

## 2. Structure of the Existing Code

### Module Organization

The `Equation` module is organized into several sub-modules and files, each handling specific aspects of equation formulation and solution:

- `equation.rs`: Defines the primary `Equation` struct and implements methods for momentum and continuity equations.
- `fields.rs`: Contains the `Fields` and `Fluxes` structs that store field variables and fluxes.
- `manager.rs`: Provides the `EquationManager` struct for managing multiple equations.
- `energy_equation.rs`: Introduces the `EnergyEquation` struct for energy conservation.
- `turbulence_models.rs`: Includes turbulence model implementations, such as the k-epsilon model.
- `reconstruction`: Handles solution reconstruction at cell faces.
- `gradient`: Manages gradient calculations.
- `flux_limiter`: Implements flux limiters to maintain numerical stability.

### Core Components

#### 2.1. Traits and Interfaces

- **`PhysicalEquation` Trait**: Defines a common interface for all physical equations, requiring an `assemble` method that builds the equation's contributions to the global system.

#### 2.2. Data Structures

- **`Fields` Struct**: Holds all field variables needed for simulations, including primary variables like pressure and additional fields for energy and turbulence.

  ```rust,ignore
  pub struct Fields {
      pub field: Section<f64>,
      pub gradient: Section<[f64; 3]>,
      pub velocity_field: Section<[f64; 3]>,
      pub temperature_field: Section<f64>,
      pub temperature_gradient: Section<[f64; 3]>,
      pub k_field: Section<f64>,
      pub epsilon_field: Section<f64>,
  }
  ```

- **`Fluxes` Struct**: Stores fluxes calculated for each equation.

  ```rust,ignore
  pub struct Fluxes {
      pub momentum_fluxes: Section<f64>,
      pub energy_fluxes: Section<f64>,
      pub turbulence_fluxes: Section<f64>,
  }
  ```

#### 2.3. Equation Implementations

- **`Equation` Struct**: Handles momentum and continuity equations, providing methods to calculate fluxes using upwinding schemes.

- **`EnergyEquation` Struct**: Manages the energy conservation equation, including thermal conductivity effects.

- **`KEpsilonModel` Struct**: Implements the k-epsilon turbulence model, calculating turbulence parameters.

#### 2.4. `EquationManager`

A central manager that holds a collection of equations to be assembled and solved.

```rust,ignore
pub struct EquationManager {
    equations: Vec<Box<dyn PhysicalEquation>>,
}
```

---

## 3. Integration of New Components

### 3.1. Expanding the `Equation` Module

The integration involves adding new structs and methods to support additional equations:

- **Energy Equation**: Represented by `EnergyEquation`, incorporating thermal effects.
- **Turbulence Models**: Implemented in `turbulence_models.rs`, starting with the k-epsilon model.

### 3.2. Implementing the `PhysicalEquation` Trait

Each new equation struct implements the `PhysicalEquation` trait, ensuring a consistent interface:

```rust,ignore
impl PhysicalEquation for EnergyEquation {
    fn assemble(
        &self,
        domain: &Mesh,
        fields: &Fields,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
    ) {
        self.calculate_energy_fluxes(
            domain,
            &fields.temperature_field,
            &fields.temperature_gradient,
            &fields.velocity_field,
            &mut fluxes.energy_fluxes,
            boundary_handler,
        );
    }
}
```

### 3.3. Modular Equation Assembly

The `EquationManager` dynamically assembles all equations added to it:

```rust,ignore
pub fn assemble_all(
    &self,
    domain: &Mesh,
    fields: &Fields,
    fluxes: &mut Fluxes,
    boundary_handler: &BoundaryConditionHandler,
) {
    for equation in &self.equations {
        equation.assemble(domain, fields, fluxes, boundary_handler);
    }
}
```

This modular approach allows for flexible inclusion or exclusion of equations based on simulation requirements.

---

## 4. Vision for Integration into the Hydra Program

### 4.1. Modularity and Extensibility

The design emphasizes modularity, enabling easy addition of new equations and models. By adhering to the `PhysicalEquation` trait, new equations can be integrated without altering existing code.

### 4.2. Dynamic Configuration

Users can configure simulations to include only the necessary equations. For instance, simulations that do not require energy conservation or turbulence modeling can omit these components, improving performance.

### 4.3. Unified Workflow

The `EquationManager` orchestrates the assembly of all equations, maintaining a unified workflow for:

- **Flux Calculations**: Using appropriate numerical schemes for each equation.
- **Gradient Computations**: Sharing gradient calculations where possible to avoid redundant computations.
- **Boundary Conditions**: Applying consistent boundary conditions across equations.

### 4.4. Scalability

The structure supports scalability to large, complex simulations by:

- Utilizing parallel computations where applicable.
- Managing resources efficiently through shared data structures like `Fields` and `Fluxes`.
- Allowing for future extensions, such as additional turbulence models or chemical species transport equations.

---

## 5. Specific Implementation Details

### 5.1. Energy Equation Integration

**`energy_equation.rs`**

- **Thermal Conductivity**: Introduced as a field in `EnergyEquation`.

  ```rust,ignore
  pub struct EnergyEquation {
      pub thermal_conductivity: f64,
  }
  ```

- **Flux Calculation**: `calculate_energy_fluxes` method computes energy fluxes similarly to momentum fluxes but includes conductive heat transfer.

### 5.2. Turbulence Model Integration

**`turbulence_models.rs`**

- **k-epsilon Model Constants**: Defined within `KEpsilonModel`.

  ```rust,ignore
  pub struct KEpsilonModel {
      pub c_mu: f64,
      pub c1_epsilon: f64,
      pub c2_epsilon: f64,
      // Additional constants and fields
  }
  ```

- **Parameter Calculation**: `calculate_turbulence_parameters` method computes turbulent kinetic energy and dissipation rate.

### 5.3. Gradient Calculations

**`gradient/gradient_calc.rs`**

- **Gradient Struct**: Manages the calculation of gradients across the mesh.

- **Boundary Condition Handling**: Incorporates boundary conditions when computing gradients, ensuring accuracy at domain boundaries.

### 5.4. Flux Reconstruction

**`reconstruction/reconstruct.rs`**

- **Linear Extrapolation**: `reconstruct_face_value` method performs linear extrapolation from cell centers to face centers.

- **Usage in Flux Calculations**: Used within flux calculation methods to estimate values at face centers, essential for accurate flux computations.

### 5.5. Flux Limiting

**`flux_limiter/flux_limiters.rs`**

- **Minmod and Superbee Limiters**: Implemented to prevent numerical oscillations and maintain Total Variation Diminishing (TVD) properties.

- **Integration in Flux Calculations**: Flux limiters are applied during flux computations to adjust reconstructed values.

### 5.6. Fields and Fluxes Management

**`fields.rs`**

- **Shared Data Structures**: `Fields` and `Fluxes` structs hold all necessary variables, reducing data duplication.

- **Extension for New Equations**: Fields specific to energy and turbulence are added, ensuring all equations have access to required data.

### 5.7. Mesh and Geometry Interactions

- **Mesh Entity Handling**: Methods are provided to retrieve cells, faces, and vertices, facilitating geometry-based calculations.

- **Geometry Calculations**: `Geometry` struct provides methods for computing volumes, areas, normals, and centroids.

### 5.8. Boundary Condition Integration

- **`BoundaryConditionHandler`**: Manages various types of boundary conditions, including Dirichlet, Neumann, and Robin conditions.

- **Application in Equations**: Boundary conditions are applied within gradient and flux calculations, ensuring they influence the solution appropriately.

---

## 6. Recommendations for Testing and Validation

### 6.1. Unit Testing

- **Coverage**: Ensure all new methods, especially in `energy_equation.rs` and `turbulence_models.rs`, have corresponding unit tests.
- **Test Cases**: Use simple, analytically solvable problems to verify correctness.

### 6.2. Integration Testing

- **Combined Equations**: Test the integration of multiple equations working together.
- **Boundary Conditions**: Validate that boundary conditions are correctly applied across different equations.

### 6.3. Benchmarking

- **Standard Benchmarks**: Compare simulation results against canonical benchmarks from *Computational Fluid Dynamics* by T.J. Chung.
  - **Laminar Flow with Heat Transfer**: Validate energy equation implementation.
  - **Turbulent Flow Over a Flat Plate**: Assess turbulence model accuracy.

### 6.4. Validation with Analytical Solutions

- **Manufactured Solutions**: Use method of manufactured solutions to test the code's ability to reproduce known solutions.
- **Error Analysis**: Perform convergence studies to verify that the numerical error decreases at the expected rate with mesh refinement.

### 6.5. Performance Testing

- **Scalability**: Test the code's performance on larger meshes to ensure efficiency.
- **Profiling**: Identify and optimize any bottlenecks, particularly in flux and gradient calculations.

---

## 7. Future Extensions

### 7.1. Additional Turbulence Models

- **Large Eddy Simulation (LES)**: Implement LES models for high-fidelity turbulence simulations.
- **Reynolds Stress Models (RSM)**: Incorporate RSM for more complex turbulence modeling.

### 7.2. Multiphysics Integration

- **Species Transport**: Add equations for chemical species to simulate reactive flows.
- **Electromagnetics**: Integrate electromagnetic equations for magnetohydrodynamics (MHD) simulations.

### 7.3. Adaptive Mesh Refinement (AMR)

- **Dynamic Mesh Refinement**: Implement AMR to improve resolution in areas with high gradients.

### 7.4. User Interface Enhancements

- **Configuration Files**: Allow users to specify simulation parameters and equations through configuration files.
- **Visualization Tools**: Integrate with visualization libraries to provide real-time feedback.

---

## Conclusion

The proposed integration plan enhances the Hydra program by extending its capabilities to solve a wider range of fluid dynamics problems, including energy conservation and turbulence modeling. The modular design ensures that the program remains flexible and maintainable, facilitating future expansions and adaptations.

By following the detailed implementation steps and recommendations provided, the Hydra program will evolve into a more powerful and versatile tool for computational fluid dynamics simulations, suitable for complex geophysical applications and beyond.

---

## References

- **Chung, T.J.** *Computational Fluid Dynamics*. Cambridge University Press, 2002.
- **HYDRA Documentation**: Refer to `about_equation.md` and `test_driven_development.md` for detailed requirements and testing procedures.

---

## Appendices

### Appendix A: Code Snippets

#### A.1. Adding an Equation to the `EquationManager`

```rust,ignore
let mut equation_manager = EquationManager::new();

// Add momentum and continuity equation
equation_manager.add_equation(Equation::new());

// Conditionally add energy equation
if simulation_requires_energy {
    equation_manager.add_equation(EnergyEquation::new(thermal_conductivity));
}

// Conditionally add turbulence model
if simulation_requires_turbulence {
    equation_manager.add_equation(KEpsilonModel::new());
}

// Assemble all equations
equation_manager.assemble_all(&mesh, &fields, &mut fluxes, &boundary_handler);
```

#### A.2. Implementing a New Turbulence Model

```rust,ignore
pub struct LESModel {
    // Fields specific to LES model
}

impl PhysicalEquation for LESModel {
    fn assemble(
        &self,
        domain: &Mesh,
        fields: &Fields,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
    ) {
        self.calculate_les_parameters(
            domain,
            &fields.velocity_field,
            &mut fluxes.turbulence_fluxes,
            boundary_handler,
        );
    }
}
```

---

### Appendix B: Testing Procedures

#### B.1. Energy Equation Validation Test

```rust,ignore
#[test]
fn test_energy_equation_fluxes() {
    // Set up mesh, fields, and boundary conditions
    let mesh = Mesh::new();
    let boundary_handler = BoundaryConditionHandler::new();
    let energy_equation = EnergyEquation::new(thermal_conductivity);

    // Initialize fields and fluxes
    let mut temperature_field = Section::new();
    let mut temperature_gradient = Section::new();
    let velocity_field = Section::new();
    let mut energy_fluxes = Section::new();

    // Populate fields with test data
    // ...

    // Call the flux calculation
    energy_equation.calculate_energy_fluxes(
        &mesh,
        &temperature_field,
        &temperature_gradient,
        &velocity_field,
        &mut energy_fluxes,
        &boundary_handler,
    );

    // Assert expected results
    // ...
}
```

---

### Appendix C: Configuration File Example

```yaml
simulation:
  equations:
    - momentum
    - energy
    - turbulence_model: k_epsilon
  parameters:
    thermal_conductivity: 0.6
    turbulence_constants:
      c_mu: 0.09
      c1_epsilon: 1.44
      c2_epsilon: 1.92
```

---

### Appendix D: Visualization Integration

Consider integrating with libraries like **ParaView** or **VisIt** for post-processing and visualization of simulation results. Output data in standard formats (e.g., VTK) to facilitate this.

---

By carefully integrating the expanded equations and following best practices in computational fluid dynamics software development, the Hydra program will provide robust and flexible simulation capabilities to users, supporting advanced research and engineering applications.