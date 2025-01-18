# Hydra `Equation` Module User Guide

---

## **Table of Contents**

1. [Introduction](#1-introduction)  
2. [Overview of the Equation Module](#2-overview-of-the-equation-module)  
3. [Core Components](#3-core-components)  
   - [PhysicalEquation Trait](#physicalequation-trait)  
   - [Fields and Fluxes](#fields-and-fluxes)  
   - [EquationManager](#equationmanager)  
4. [Equation Submodules](#4-equation-submodules)  
   - [MomentumEquation](#momentumequation)  
   - [EnergyEquation](#energyequation)  
   - [Turbulence Models](#turbulence-models)  
   - [Flux Limiters](#flux-limiters)  
   - [Gradient Computation](#gradient-computation)  
   - [Reconstruction Methods](#reconstruction-methods)  
5. [Using the Equation Module](#5-using-the-equation-module)  
   - [Defining a Physical Equation](#defining-a-physical-equation)  
   - [Managing Equations, Fields, and Fluxes](#managing-equations-fields-and-fluxes)  
   - [Integration with Other Modules](#integration-with-other-modules)  
6. [Best Practices](#6-best-practices)  
7. [Conclusion](#7-conclusion)

---

## **1. Introduction**

The **`Equation`** module in Hydra coordinates the **physical equations** governing fluid or other physical processes. It ties together:

- **Equation Definitions**: (e.g., **Momentum**, **Energy**, **Turbulence**).  
- **Fields**: Data structures storing state variables like velocity, pressure, temperature.  
- **Fluxes**: Computed results from each equation used to update the fields.  
- **Reconstruction**, **Gradients**, and **Flux Limiters**: Tools for higher-order accuracy and numerical stability.

By leveraging the **`PhysicalEquation`** trait, the module supports adding new physics while tapping into Hydra’s frameworks for domain geometry, boundary conditions, and solver time-stepping.

---

## **2. Overview of the Equation Module**

**Location**: `src/equation/`

Submodules:

- **`equation/fields.rs`**: Contains `Fields` (various field data) and `Fluxes` (various flux data) plus the `UpdateState` trait.  
- **`equation/equation.rs`**: An example utility class for flux calculations.  
- **`equation/manager.rs`**: The `EquationManager` that orchestrates multiple equations with time stepping.  
- **`energy_equation.rs`, `momentum_equation.rs`, `turbulence_models.rs`**: Implementation examples of physical equations.  
- **`flux_limiter`, `gradient`, `reconstruction`**: Supporting tools for advanced numerical methods.

---

## **3. Core Components**

### PhysicalEquation Trait

```rust
pub trait PhysicalEquation {
    fn assemble(
        &self,
        domain: &Mesh,
        fields: &Fields,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
        current_time: f64,
    );
}
```

Defines a method `assemble(...)` that modifies or adds to **fluxes** based on **fields** and domain data. It's the core interface for PDE-based computations:

1. **domain**: The mesh describing geometry.  
2. **fields**: Current field data (e.g., velocity, pressure).  
3. **fluxes**: Outgoing flux container updated by the equation.  
4. **boundary_handler**: For applying boundary conditions.  
5. **current_time**: For time-dependent BC or physics.

**Implementers** might:

- Calculate momentum fluxes (momentum_equation).
- Calculate energy fluxes (energy_equation).
- Or other physical processes (turbulence models, chemical reactions, etc.).

### Fields and Fluxes

In **`fields.rs`**:

- **`Fields`**: Contains multiple **scalar_fields**, **vector_fields**, **tensor_fields** (backed by Hydra’s `Section<T>`).
  - `get_scalar_field_value(...)`, `set_scalar_field_value(...)`, similarly for vectors.  
  - `update_from_fluxes(...)`: A convenient method to apply flux changes to the fields.

- **`Fluxes`**: Gathers flux data (e.g., momentum_fluxes, energy_fluxes) in `Section<T>` objects.  
  - e.g., `add_momentum_flux(...)`, `add_energy_flux(...)`, `add_turbulence_flux(...)`.

**`UpdateState`** trait: Let fields do a “time step update” (`update_state`), compute difference, or measure norm.

### EquationManager

In **`manager.rs`**:

```rust
pub struct EquationManager {
    equations: Vec<Box<dyn PhysicalEquation>>,
    time_stepper: Box<dyn TimeStepper<Self>>,
    domain: Arc<RwLock<Mesh>>,
    boundary_handler: Arc<RwLock<BoundaryConditionHandler>>,
}
```

- Maintains a list of `PhysicalEquation`s.  
- Has a reference to a time-stepper (which uses it as a `TimeDependentProblem`):
  - `EquationManager` itself implements `TimeDependentProblem` so it can produce flux derivatives for time stepping.  
- **`assemble_all(...)`**: Runs `assemble(...)` for each equation over `fields -> fluxes`.  
- **`step(...)`**: Delegates a time-step to the assigned `time_stepper`, which calls back into `compute_rhs(...)` etc.

---

## **4. Equation Submodules**

### MomentumEquation

File: **`momentum_equation.rs`**  
Implements `PhysicalEquation` for incompressible (or general) momentum laws:

- **`assemble(...)`** calls `calculate_momentum_fluxes(...)`:
  - Collect velocity, pressure data from `fields`.
  - Compute fluxes for convection, diffusion, pressure.  
  - Apply BC adjustments.  
  - Store final flux in `fluxes.momentum_fluxes`.  

It’s a template for more advanced Navier-Stokes or specialized momentum systems.

### EnergyEquation

File: **`energy_equation.rs`**  
Implements `PhysicalEquation` for thermal energy or enthalpy equation:

- **`assemble(...)`** calls `calculate_energy_fluxes(...)`:
  - Evaluates conduction + convection flux in a manner similar to momentum.
  - BCs can impose temperature constraints or flux.  

### Turbulence Models

- **`turbulence_models.rs`**: A trait `TurbulenceModel` plus an example `GOTMModel`.
- They typically define how to compute **eddy diffusivity**, **eddy viscosity**, or extra scalar fluxes (k-ε, RANS, etc.).

**Note**: They also implement `PhysicalEquation` if they produce fluxes for e.g. TKE or dissipation.

### Flux Limiters

Directory: **`flux_limiter/`**  
- Defines a `FluxLimiter` trait with implementations: **Minmod**, **Superbee**, **VanLeer**, etc.  
- Typically used in reconstructions or slope-limited finite volume methods to maintain stability near discontinuities.

### Gradient Computation

Directory: **`gradient/`**  
- **`GradientMethod`** trait with **`FiniteVolumeGradient`** and **`LeastSquaresGradient`**.  
- A `Gradient` wrapper that calls these methods for each cell.  
- Allows the module to compute \(\nabla \phi\) (scalar gradient) for advanced flux computations or higher-order PDE solvers.

### Reconstruction Methods

Directory: **`reconstruction/`**  
- **`ReconstructionMethod`** trait for face reconstruction.  
- Implementations:
  - **`LinearReconstruction`** (simple linear interpolation).  
  - **`WENOReconstruction`**, **`PPMReconstruction`**, etc. for higher-order schemes.  
- The PDE solvers (like momentum/energy) can call these to get face values from cell-centered data plus gradients.

---

## **5. Using the Equation Module**

### Defining a Physical Equation

1. **Implement** `PhysicalEquation`:
   ```rust
   pub struct MyEquation;

   impl PhysicalEquation for MyEquation {
       fn assemble(
           &self,
           domain: &Mesh,
           fields: &Fields,
           fluxes: &mut Fluxes,
           boundary_handler: &BoundaryConditionHandler,
           current_time: f64,
       ) {
           // 1) Possibly compute gradient or field reconstructions.
           // 2) Evaluate fluxes (convective, diffusive, source).
           // 3) Use boundary_handler to apply BC modifications.
           // 4) Insert final flux into fluxes (e.g. fluxes.momentum_fluxes).
       }
   }
   ```

2. Add custom logic for your PDE or equations. The momentum and energy equation files are good references.

### Managing Equations, Fields, and Fluxes

- **Create** an `EquationManager` with domain, boundary handler, and time stepper:
  ```rust
  let manager = EquationManager::new(
      time_stepper,  // e.g., a Box<dyn TimeStepper<EquationManager>>
      domain,
      boundary_handler,
  );
  ```
- **Add** `PhysicalEquation` objects:
  ```rust
  manager.add_equation(MomentumEquation::new());
  manager.add_equation(EnergyEquation::new(thermal_conductivity));
  ```
- **Fields**: Usually start with an initial `Fields` object.  
- **Time stepping**: If the manager is used in synergy with Hydra’s time stepping, call `manager.step(&mut fields)` each iteration. This internally calls `EquationManager::compute_rhs(...)` which does `assemble_all(...)` -> fluxes -> derivative.

### Integration with Other Modules

- **Mesh**: The domain geometry from **`domain`** module.  
- **Boundary Conditions**: The **`BoundaryConditionHandler`** from **`boundary`** module.  
- **Time Stepping**: The **`TimeStepper`** trait from **`time_stepping`**.  
- **Linear Algebra**: If a PDE step requires a matrix solve, call `solve_linear_system(...)` with Hydra’s `Matrix` interface.

---

## **6. Best Practices**

1. **Keep PDE Logic in `PhysicalEquation`**: Encapsulate each PDE’s flux computations in a separate struct.  
2. **Use `EquationManager`**: Orchestrate multiple PDEs (momentum, energy, turbulence) to produce combined flux data.  
3. **Extend with `FluxLimiter`**: If high-order or slope-limited finite volume methods are used, incorporate limiters in your reconstruction steps.  
4. **Check Boundary Conditions**: The boundary handler must provide the correct BC for each face or cell boundary.  
5. **Parallelization**: Hydra’s module design allows concurrency at the level of flux assembly if carefully managed.  

---

## **7. Conclusion**

The **`Equation`** module is a crucial piece of Hydra’s PDE-solving architecture:

- **`PhysicalEquation`** provides a uniform interface for PDE flux assembly.  
- **`Fields`** and **`Fluxes`** store, update, and unify the data needed for continuous PDE integration over the mesh.  
- **`EquationManager`** merges these equations with time stepping, letting Hydra evolve complex multi-physics systems in a consistent manner.  
- **Supporting submodules** (momentum, energy, turbulence, reconstruction, flux limiting, gradient) demonstrate extensible approaches for real-world fluid or multi-physics simulations.

By following this modular approach, you can **add new physics**, **customize flux calculations**, or **switch reconstruction/gradient methods** to tailor Hydra to your numerical requirements.