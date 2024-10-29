### Extrusion Module Documentation Outline for the Hydra Project

The `extrusion` module is responsible for transforming 2D geophysical meshes into 3D representations. It provides a structured and extendable system to handle extrusion of various cell types, such as quadrilaterals and triangles, into corresponding 3D volumetric cells (hexahedrons and prisms). This module is divided into four main sections, detailed below, each contributing a specific responsibility to the extrusion pipeline.

---

#### 1. **Core (`core`)**
   - **Overview**: Defines fundamental data structures (`QuadrilateralMesh`, `TriangularMesh`) and an extrusion-oriented trait (`ExtrudableMesh`).
   - **Components**:
     - **`extrudable_mesh`**: Provides the `ExtrudableMesh` trait, ensuring all meshes implement core methods for extrusion (e.g., type-checking and retrieval of vertices/cells).
     - **`hexahedral_mesh`**: Implements the `QuadrilateralMesh` structure, designed for extrusion into hexahedral cells.
     - **`prismatic_mesh`**: Implements the `TriangularMesh` structure, designed for extrusion into prismatic cells.

#### 2. **Infrastructure (`infrastructure`)**
   - **Overview**: Manages input/output operations for the module, including loading and saving mesh files, and provides logging capabilities.
   - **Components**:
     - **`mesh_io`**: Provides the `MeshIO` struct, handling loading of 2D Gmsh mesh files, and saving extruded 3D meshes.
     - **`logger`**: Contains a basic logging implementation, facilitating logging of information, warnings, and errors during extrusion operations.

#### 3. **Interface Adapters (`interface_adapters`)**
   - **Overview**: Acts as the main interface for external modules or services to interact with the extrusion functionalities.
   - **Components**:
     - **`extrusion_service`**: The primary interface (`ExtrusionService`) that interprets mesh types and invokes extrusion operations, determining the correct 3D extrusion form (hexahedral or prismatic).

#### 4. **Use Cases (`use_cases`)**
   - **Overview**: Implements specific operations necessary for extrusion, organizing the process into reusable functions for both vertices and cells.
   - **Components**:
     - **`vertex_extrusion`**: Provides methods for extruding vertices along the z-axis to create 3D layers from a 2D base layer.
     - **`cell_extrusion`**: Contains logic for extruding 2D cells (quadrilaterals and triangles) into 3D volumes (hexahedrons and prisms).
     - **`extrude_mesh`**: Combines `vertex_extrusion` and `cell_extrusion` into complete mesh extrusion methods, converting a full 2D mesh into a 3D representation.

---

### Critical Findings in the Module

The failing tests indicate potential issues primarily within the `MeshIO::load_2d_mesh` function and its handling of specific cell configurations. Here are a few key points to address:

1. **Unsupported Cell Types**: 
   - **Issue**: The `load_2d_mesh` function currently fails when encountering cells that don’t strictly match quadrilateral or triangular configurations. The `GmshParser` may be detecting unexpected cell configurations.
   - **Solution**: Refine the cell verification process within `load_2d_mesh` to ensure it gracefully handles and logs unsupported cell types while proceeding with valid cells.

2. **Downcasting Failures in `ExtrusionService`**:
   - **Issue**: Some tests show downcasting errors when attempting to extrude meshes, suggesting `ExtrusionService` may be handling certain meshes incorrectly.
   - **Solution**: Add error handling within `ExtrusionService` to catch and report downcasting issues, ensuring meshes are fully validated before extrusion attempts.

3. **Layer and Depth Handling in `VertexExtrusion`**:
   - **Issue**: Tests for zero-layer extrusion suggest that edge cases for `depth` and `layers` might be unhandled.
   - **Solution**: Add validation to ensure meaningful results even with unconventional input values (e.g., `layers = 0`).

---

#### Section 1: Core (`core`)

**Overview**  
The `core` module defines the foundational data structures and traits necessary for the extrusion process. This section introduces the `ExtrudableMesh` trait, which standardizes the properties and behaviors required for 2D meshes to be extruded into 3D representations. The module also includes implementations of specific 2D mesh types (`QuadrilateralMesh` and `TriangularMesh`), each designed for extrusion into distinct 3D volumetric structures (hexahedrons and prisms, respectively).

---

#### Module Structure

- **`extrudable_mesh`**  
- **`hexahedral_mesh`**  
- **`prismatic_mesh`**  

---

#### 1.1 `ExtrudableMesh` Trait (extrudable_mesh.rs)

The `ExtrudableMesh` trait establishes a common interface for 2D mesh structures that support extrusion. This trait provides:
- **Mesh Validity**: Ensures the mesh can be extruded by verifying the cell structure (quadrilateral or triangular).
- **Mesh Type Identification**: Methods for determining if the mesh is quadrilateral or triangular.
- **Data Retrieval**: Access to the vertices and cells of the mesh, facilitating the extrusion process.

##### Trait Methods

- `is_valid_for_extrusion(&self) -> bool`:  
  Verifies the mesh's cells conform to the expected structure (all quadrilateral or all triangular). This ensures compatibility with extrusion routines.
  
- `get_vertices(&self) -> Vec<[f64; 3]>`:  
  Retrieves the 3D coordinates of the mesh vertices, formatted for extrusion.

- `get_cells(&self) -> Vec<Vec<usize>>`:  
  Returns the cell-to-vertex connectivity, where each cell is represented by vertex indices into the `vertices` array.

- **Type Identification Methods**:  
  - `is_quad_mesh(&self) -> bool`: Returns `true` if the mesh is quadrilateral.
  - `is_tri_mesh(&self) -> bool`: Returns `true` if the mesh is triangular.

- **Type Downcasting**:
  - `as_quad(&self) -> Option<&QuadrilateralMesh>`: Attempts to downcast the mesh to a `QuadrilateralMesh`.
  - `as_tri(&self) -> Option<&TriangularMesh>`: Attempts to downcast the mesh to a `TriangularMesh`.

---

#### 1.2 `QuadrilateralMesh` Structure (hexahedral_mesh.rs)

The `QuadrilateralMesh` struct represents a 2D mesh composed of quadrilateral cells, extrudable into 3D hexahedral cells. This structure is designed to handle meshes consisting of quadrilaterals exclusively, ensuring compatibility with hexahedral extrusion processes.

##### Key Elements

- **Vertices**:  
  Stored as `Vec<[f64; 3]>`, where each entry represents a vertex in 3D space (x, y, z = 0).
  
- **Cells**:  
  Stored as `Vec<Vec<usize>>`, with each inner vector containing 4 indices into the `vertices` array, representing a quadrilateral cell.

##### Methods

- **Constructor**:  
  `new(vertices: Vec<[f64; 3]>, cells: Vec<Vec<usize>>) -> Self`:  
  Initializes the mesh with the given vertices and cells.

- **ExtrudableMesh Implementation**:
  - `is_valid_for_extrusion(&self) -> bool`: Checks that each cell has exactly 4 vertices.
  - `get_vertices`, `get_cells`, `as_any`: Retrieves data and provides type erasure support.

##### Example Usage

```rust
let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
let cells = vec![vec![0, 1, 2, 3]];
let quad_mesh = QuadrilateralMesh::new(vertices, cells);
assert!(quad_mesh.is_valid_for_extrusion());
```

---

#### 1.3 `TriangularMesh` Structure (prismatic_mesh.rs)

The `TriangularMesh` struct represents a 2D mesh composed of triangular cells, intended for extrusion into 3D prismatic cells. This structure supports meshes that consist exclusively of triangular cells, ensuring compatibility with prismatic extrusion routines.

##### Key Elements

- **Vertices**:  
  Stored as `Vec<[f64; 3]>`, where each entry represents a vertex in 3D space (x, y, z = 0).

- **Cells**:  
  Stored as `Vec<Vec<usize>>`, with each inner vector containing 3 indices into the `vertices` array, representing a triangular cell.

##### Methods

- **Constructor**:  
  `new(vertices: Vec<[f64; 3]>, cells: Vec<Vec<usize>>) -> Self`:  
  Initializes the mesh with the given vertices and cells.

- **ExtrudableMesh Implementation**:
  - `is_valid_for_extrusion(&self) -> bool`: Checks that each cell has exactly 3 vertices.
  - `get_vertices`, `get_cells`, `as_any`: Retrieves data and provides type erasure support.

##### Example Usage

```rust
let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];
let cells = vec![vec![0, 1, 2]];
let tri_mesh = TriangularMesh::new(vertices, cells);
assert!(tri_mesh.is_valid_for_extrusion());
```

---

#### Summary

The `core` module defines the structural and behavioral requirements for meshes in the extrusion process. By standardizing mesh validation and access, the `ExtrudableMesh` trait enables flexible and consistent extrusion across different 2D cell types, supporting complex 3D mesh generation. The `QuadrilateralMesh` and `TriangularMesh` structures implement this trait, ensuring they are readily usable within extrusion workflows for hexahedral and prismatic cell types, respectively.

---

### Section 2: Infrastructure (`infrastructure`)

**Overview**  
The `infrastructure` module handles input/output and logging capabilities essential for the extrusion module’s operations. This module includes `MeshIO` for file-based interactions with mesh data and `Logger` for capturing operational logs. These utilities support efficient loading, saving, and diagnostic logging, facilitating troubleshooting and mesh management within the extrusion workflows.

---

#### Module Structure

- **`mesh_io`**  
- **`logger`**

---

#### 2.1 `MeshIO` Structure (mesh_io.rs)

The `MeshIO` structure is responsible for reading and writing mesh data, enabling interaction with mesh files in the Gmsh format. This structure supports loading 2D mesh files for extrusion and saving 3D extruded meshes.

##### Key Methods

- **`load_2d_mesh(file_path: &str) -> Result<Box<dyn ExtrudableMesh>, String>`**  
  Loads a 2D mesh from a Gmsh file. The function identifies the mesh type (quadrilateral or triangular) based on cell configuration, constructing either a `QuadrilateralMesh` or a `TriangularMesh`. This operation fails if the file contains unsupported cell types.

  - **Parameters**:
    - `file_path`: Path to the Gmsh file containing 2D mesh data.
  
  - **Returns**:  
    A boxed `ExtrudableMesh` instance (either `QuadrilateralMesh` or `TriangularMesh`) if successful, or an error message if loading fails due to unsupported cell types or file issues.

  - **Error Handling**:  
    The function returns specific error messages for:
    - Unsupported cell configurations
    - Mesh validation failures
  
  - **Example Usage**:
    ```rust
    let mesh = MeshIO::load_2d_mesh("inputs/rectangle_quad.msh2")?;
    assert!(mesh.is_quad_mesh());
    ```

- **`save_3d_mesh(mesh: &Mesh, file_path: &str) -> Result<(), String>`**  
  Saves an extruded 3D mesh to a file in a Gmsh-compatible format. This function writes vertices and cells to the specified file.

  - **Parameters**:
    - `mesh`: Reference to the `Mesh` containing the 3D mesh data.
    - `file_path`: Path to save the file.
  
  - **Returns**:  
    `Ok(())` if the save operation is successful, or an error message if it fails due to I/O issues.

  - **Error Handling**:  
    Returns an error message if the file cannot be created or written to.

  - **Example Usage**:
    ```rust
    let mesh = Mesh::new();
    MeshIO::save_3d_mesh(&mesh, "outputs/extruded_mesh.msh")?;
    ```

##### Internal Helper Methods in `MeshIO`

- **`get_vertices`**  
  Retrieves all vertices from a 3D mesh, formatted as a vector of `[f64; 3]` coordinates.

- **`get_cell_vertex_indices`**  
  Returns vertex indices for all cells, useful for writing cell definitions in Gmsh format.

---

#### 2.2 `Logger` Structure (logger.rs)

The `Logger` structure provides configurable logging functionality for tracking operations within the extrusion process. The logger supports multiple logging levels (info, warning, error) with timestamped messages, improving traceability and debugging.

##### Key Methods

- **`new(file_path: Option<&str>) -> Result<Self, io::Error>`**  
  Initializes a new logger instance. If `file_path` is provided, logs are written to the specified file; otherwise, logs are directed to stdout.

  - **Parameters**:
    - `file_path`: Optional file path for log output.
  
  - **Returns**:  
    A new `Logger` instance or an `io::Error` if file creation fails.

  - **Example Usage**:
    ```rust
    let mut logger = Logger::new(Some("logs/extrusion.log"))?;
    ```

- **`info(&mut self, message: &str)`**  
  Logs an informational message with a timestamp, useful for standard process updates.

- **`warn(&mut self, message: &str)`**  
  Logs a warning message, signaling potential issues in the extrusion workflow.

- **`error(&mut self, message: &str)`**  
  Logs an error message, ideal for tracking critical failures or unexpected conditions.

- **Core Logging Function**:  
  All log methods utilize the `log` method to format messages with timestamps and log levels.

---

#### Summary

The `infrastructure` module provides essential utilities to support the core extrusion workflows. `MeshIO` enables efficient file handling for both 2D and 3D meshes, supporting seamless integration with external files. `Logger` tracks the module’s operations, capturing critical information, warnings, and errors, which aids in debugging and operational transparency. Together, these components ensure robust file I/O and traceability within the extrusion processes.

---

### Section 3: Interface Adapters (`interface_adapters`)

**Overview**  
The `interface_adapters` module provides an interface for invoking extrusion processes in a user-friendly manner, encapsulating complex logic into straightforward API calls. The primary component, `ExtrusionService`, acts as the interface for extruding a 2D mesh into a 3D representation, determining the appropriate extrusion method based on mesh type (quadrilateral or triangular).

---

#### Module Structure

- **`extrusion_service`**

---

#### 3.1 `ExtrusionService` (extrusion_service.rs)

The `ExtrusionService` struct is the main entry point for initiating the extrusion process. It simplifies interactions with the underlying mesh types and extrusion use cases, ensuring that users can effortlessly extrude both quadrilateral and triangular meshes. This module provides a high-level API to extrude meshes into 3D structures and includes error handling to report issues such as unsupported mesh types.

---

##### Key Method

- **`extrude_mesh(mesh: &dyn ExtrudableMesh, depth: f64, layers: usize) -> Result<Mesh, String>`**  
  This method serves as the primary extrusion function, extruding a 2D mesh into a 3D representation. It first identifies the type of the given mesh (quadrilateral or triangular) and then calls the appropriate extrusion use case (`ExtrudeMeshUseCase::extrude_to_hexahedron` for quadrilateral meshes or `ExtrudeMeshUseCase::extrude_to_prism` for triangular meshes).

  - **Parameters**:
    - `mesh`: A reference to a 2D mesh implementing the `ExtrudableMesh` trait.
    - `depth`: The extrusion depth along the z-axis.
    - `layers`: The number of layers for the extrusion, defining the mesh's vertical resolution.
  
  - **Returns**:  
    `Ok(Mesh)` if extrusion is successful, containing the fully extruded 3D mesh, or an error message if the extrusion fails.

  - **Error Handling**:  
    Returns a string error message if the mesh type is unsupported or if type-casting to `QuadrilateralMesh` or `TriangularMesh` fails.

  - **Example Usage**:
    ```rust
    let quad_mesh = QuadrilateralMesh::new(...);
    let depth = 5.0;
    let layers = 3;
    let result = ExtrusionService::extrude_mesh(&quad_mesh, depth, layers);
    assert!(result.is_ok());
    ```

##### Mesh Type Detection and Downcasting

The `extrude_mesh` method first checks if the mesh is quadrilateral or triangular using `is_quad_mesh` and `is_tri_mesh` methods from the `ExtrudableMesh` trait. Once identified, it attempts to downcast the `ExtrudableMesh` trait object to the specific type (e.g., `QuadrilateralMesh`) and performs the extrusion.

##### Example Code for Quadrilateral Mesh

```rust
let quad_mesh = QuadrilateralMesh::new(
    vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
    vec![vec![0, 1, 2, 3]],
);
let extruded_mesh = ExtrusionService::extrude_mesh(&quad_mesh, 3.0, 2);
assert!(extruded_mesh.is_ok(), "Extrusion should succeed for quadrilateral mesh");
```

---

#### Summary

The `interface_adapters` module provides an intuitive API for triggering mesh extrusion operations, abstracting the complexities of identifying and extruding specific mesh types. By handling mesh type detection, downcasting, and use-case invocation, `ExtrusionService` empowers users to work seamlessly with both quadrilateral and triangular meshes in 2D, transforming them into well-structured 3D representations. The encapsulated error handling and type-specific processing simplify integration and improve error transparency, making the extrusion API robust and user-friendly.

---

### Section 4: Use Cases (`use_cases`)

**Overview**  
The `use_cases` module organizes the core logic for the extrusion process, breaking down the 2D-to-3D extrusion into focused, manageable components. This module provides structures and functions for extruding both vertices and cells, enabling the extrusion of either quadrilateral or triangular meshes into three-dimensional hexahedral or prismatic forms. The primary entry point is the `ExtrudeMeshUseCase`, which coordinates vertex and cell extrusion to create the final 3D mesh structure.

---

#### Module Structure

- **`cell_extrusion`**: Defines logic for extruding 2D cell structures (quadrilateral and triangular) into 3D cells.
- **`extrude_mesh`**: Orchestrates the overall extrusion process, generating 3D meshes from 2D inputs.
- **`vertex_extrusion`**: Handles the extrusion of vertices to create multiple layers in the z-direction.

---

#### 4.1 `CellExtrusion` (cell_extrusion.rs)

The `CellExtrusion` struct provides methods to extrude 2D cells (quadrilaterals or triangles) into 3D volumetric cells (hexahedrons or prisms) across multiple layers. 

- **`extrude_quadrilateral_cells(cells: Vec<Vec<usize>>, layers: usize) -> Vec<Vec<usize>>`**  
  This function extrudes each quadrilateral cell into a hexahedron across multiple layers, creating eight vertices per cell. It iterates over each layer, calculating the vertex indices for each layer to construct the hexahedron’s vertices.

  - **Parameters**:
    - `cells`: Vector of quadrilateral cells, each defined by four vertex indices.
    - `layers`: Number of layers for extrusion.
  
  - **Returns**:  
    A vector of hexahedral cells, where each cell has eight vertices.

- **`extrude_triangular_cells(cells: Vec<Vec<usize>>, layers: usize) -> Vec<Vec<usize>>`**  
  Extrudes triangular cells into prismatic cells across layers, where each prism has six vertices. Similar to quadrilateral cells, this method calculates each layer’s vertex indices for a prismatic structure.

  - **Parameters**:
    - `cells`: Vector of triangular cells, each defined by three vertex indices.
    - `layers`: Number of layers for extrusion.
  
  - **Returns**:  
    A vector of prismatic cells, each defined by six vertices.

---

#### 4.2 `ExtrudeMeshUseCase` (extrude_mesh.rs)

`ExtrudeMeshUseCase` is responsible for orchestrating the entire extrusion process, including extruding vertices and constructing the final 3D mesh from cells. The two main methods, `extrude_to_hexahedron` and `extrude_to_prism`, handle quadrilateral and triangular meshes respectively.

- **`extrude_to_hexahedron(mesh: &QuadrilateralMesh, depth: f64, layers: usize) -> Result<Mesh, String>`**  
  Extrudes a quadrilateral 2D mesh into a 3D mesh with hexahedral cells by:
  1. Extruding the vertices using `VertexExtrusion`.
  2. Extruding the quadrilateral cells to hexahedrons using `CellExtrusion`.
  3. Assembling these vertices and cells into a `Mesh` object.

  - **Parameters**:
    - `mesh`: Reference to the `QuadrilateralMesh` being extruded.
    - `depth`: Total depth of extrusion.
    - `layers`: Number of layers for extrusion.
  
  - **Returns**:  
    `Ok(Mesh)` with the fully extruded mesh on success or an error message if the mesh is invalid.

- **`extrude_to_prism(mesh: &TriangularMesh, depth: f64, layers: usize) -> Result<Mesh, String>`**  
  Similar to `extrude_to_hexahedron`, but operates on triangular meshes, producing prismatic cells.

---

#### 4.3 `VertexExtrusion` (vertex_extrusion.rs)

The `VertexExtrusion` struct manages the extrusion of vertices, creating layers of vertices along the z-axis to provide a foundation for constructing 3D cells. Vertices from each layer serve as references for corresponding cells in the extrusion.

- **`extrude_vertices(vertices: Vec<[f64; 3]>, depth: f64, layers: usize) -> Vec<[f64; 3]>`**  
  Creates layers of vertices by shifting the z-coordinate for each layer, with the depth evenly divided among the layers.

  - **Parameters**:
    - `vertices`: List of base vertices (z = 0).
    - `depth`: Total extrusion depth.
    - `layers`: Number of layers to create.
  
  - **Returns**:  
    Vector of extruded vertices, with each layer positioned at incremental z-values.

---

#### Summary

The `use_cases` module provides a robust, modular foundation for mesh extrusion by decoupling the logic for vertex and cell extrusion and orchestrating these steps to form complete 3D meshes. This layered approach allows the `ExtrudeMeshUseCase` to perform high-level extrusion tasks with precise control over the type of cells generated, making it adaptable for different types of 2D meshes.

### Test Failure Analysis Report for Extrusion Module

This report covers the recent test failures in the `extrusion` module based on the provided test failure logs and a detailed review of the module's source code. The failures are in `mesh_io` and various extrusion tests, which involve loading and validating quadrilateral and triangular meshes, as well as performing 2D-to-3D extrusion.

#### Summary of Failed Tests
1. `extrusion::infrastructure::mesh_io::tests::test_load_quadrilateral_mesh`
2. `extrusion::infrastructure::mesh_io::tests::test_load_triangular_mesh`
3. `extrusion::tests::tests::test_extrude_rectangle_to_hexahedron`
4. `extrusion::tests::tests::test_extrude_triangle_basin_to_prism`
5. `extrusion::tests::tests::test_extrusion_service_for_hexahedron`

---

### Failure 1: `test_load_quadrilateral_mesh`

- **Error Message**: "Unsupported cell type: cells must be either quadrilateral or triangular."
- **Location**: `mesh_io.rs`, line 189
- **Expected Outcome**: Successful loading of a quadrilateral mesh

#### Analysis
The `MeshIO::load_2d_mesh` function checks each cell's vertex count to determine its type: quadrilateral (4 vertices) or triangular (3 vertices). The error indicates that the cell type in the mesh file provided is not recognized as quadrilateral, triangular, or possibly has inconsistent vertex counts. The following potential causes are identified:
   - **Mesh File Integrity**: The mesh file may contain cells that do not strictly adhere to quadrilateral or triangular configurations.
   - **Cell Validation Logic**: A mismatch in cell type detection logic could misidentify cells with unexpected configurations or corrupted vertex indexing.

#### Recommendation
1. **Validate Mesh File Content**: Confirm the mesh file used (`rectangular_channel_quad.msh2`) only contains quadrilateral cells with exactly 4 vertices each.
2. **Enhance Logging**: Add debugging logs within the `load_2d_mesh` function to output details of any cells that fail validation.
3. **Cell Count Check**: Modify the validation logic to account for extraneous or missing vertices in cells that should be quadrilateral.

---

### Failure 2: `test_load_triangular_mesh`

- **Error Message**: Unexpected failure to load a triangular mesh.
- **Location**: `mesh_io.rs`, line 200
- **Expected Outcome**: Successful loading and recognition of a triangular mesh

#### Analysis
This failure suggests that `load_2d_mesh` did not recognize the triangular mesh or encountered an issue when validating triangular cells. Since `load_2d_mesh` requires all cells to have a consistent type (either all quadrilateral or all triangular), this could result from:
   - **Mixed Cell Types**: The mesh file (`triangular_basin.msh2`) might contain a mixture of cell types, which would not be compatible with the current validation logic.
   - **Incorrect Vertex Counts**: If any cells within the file contain more or fewer than three vertices, they will not be recognized as triangles.

#### Recommendation
1. **Mesh File Consistency**: Verify that `triangular_basin.msh2` exclusively contains cells with exactly 3 vertices each.
2. **Error Logging for Unsupported Cells**: Modify the error handling to log details about cells that fail to meet the triangular criteria.

---

### Failure 3: `test_extrude_rectangle_to_hexahedron`

- **Error Message**: `assertion failed: extruded_mesh.is_ok()`
- **Location**: `tests.rs`, line 24
- **Expected Outcome**: Successful extrusion of a quadrilateral mesh into a hexahedral 3D mesh

#### Analysis
The extrusion operation fails at the point of validation for successful mesh creation. This likely indicates:
   - **Invalid Quadrilateral Mesh Structure**: If `extrude_to_hexahedron` detects invalid configurations in the quadrilateral mesh, it will not perform the extrusion.
   - **Inconsistent Vertex or Cell Counting**: The final vertex and cell counts expected from extrusion may not align with the actual counts generated, leading to a validation failure.

#### Recommendation
1. **Verify Quadrilateral Mesh Validity**: Before extrusion, confirm that the quadrilateral mesh contains valid, non-degenerate cells.
2. **Count and Dimension Checks**: Add checks within `extrude_to_hexahedron` to log expected versus actual vertex and cell counts, ensuring correct 3D cell assembly.

---

### Failure 4: `test_extrude_triangle_basin_to_prism`

- **Error Message**: `assertion failed: extruded_mesh.is_ok()`
- **Location**: `tests.rs`, line 60
- **Expected Outcome**: Successful extrusion of a triangular mesh into a prismatic 3D mesh

#### Analysis
Similar to the previous extrusion test, this failure arises from a failure in the `extrude_to_prism` process. Since `extrude_to_prism` relies on:
   - **Valid Triangle Structure**: Ensuring that all triangular cells in the mesh contain exactly 3 vertices.
   - **Layering and Vertex Consistency**: Each triangular cell should result in a prismatic cell that maintains vertex integrity across layers.

#### Recommendation
1. **Triangle Integrity Check**: Ensure that all cells contain exactly 3 vertices.
2. **Enhance `extrude_to_prism` Debugging**: Include additional logs to capture any discrepancies in vertex indexing or unexpected layer configurations.

---

### Failure 5: `test_extrusion_service_for_hexahedron`

- **Error Message**: `assertion failed: extruded_mesh.is_ok()`
- **Location**: `tests.rs`, line 96
- **Expected Outcome**: Successful extrusion via the `ExtrusionService`

#### Analysis
This failure indicates that `ExtrusionService::extrude_mesh`, which abstracts the extrusion of both quadrilateral and triangular meshes, did not successfully perform extrusion for a quadrilateral mesh. Likely issues include:
   - **Mesh Type Misidentification**: The `ExtrusionService` may incorrectly determine the mesh type, failing to cast to `QuadrilateralMesh`.
   - **Inconsistent Mesh Data**: If the mesh data is malformed or contains unsupported cells, extrusion will fail.

#### Recommendation
1. **Log Mesh Type Detection**: Add logging within `ExtrusionService::extrude_mesh` to verify mesh type detection.
2. **Handle Invalid Mesh Scenarios**: Improve error handling to differentiate failures due to type mismatch from those caused by other structural issues.

---

### General Recommendations

- **Enhanced Logging**: Across all relevant functions (`load_2d_mesh`, `extrude_to_hexahedron`, `extrude_to_prism`, `extrude_mesh`), add logging to capture:
  - Cell vertex counts per type (triangular or quadrilateral)
  - Mesh type and structure validation steps
  - Expected versus actual vertex and cell counts post-extrusion

- **Unit Testing of Cell Validity**: Integrate additional unit tests to validate cell consistency (i.e., all quadrilateral or triangular) before the full extrusion process.

- **Consistent Mesh File Verification**: Ensure the mesh files used (`rectangular_channel_quad.msh2`, `triangular_basin.msh2`) conform strictly to the expected quadrilateral and triangular configurations without any inconsistencies.

These steps should improve traceability and help isolate structural or configuration-based errors within the extrusion pipeline.

Here is the source code tree:
```bash
C:.
│   mod.rs
│   tests.rs
│
├───core
│       extrudable_mesh.rs
│       hexahedral_mesh.rs
│       mod.rs
│       prismatic_mesh.rs
│
├───docs
│       about_extrusion.md
│       gp.md
│
├───infrastructure
│       logger.rs
│       mesh_io.rs
│       mod.rs
│
├───interface_adapters
│       extrusion_service.rs
│       mod.rs
│
└───use_cases
        cell_extrusion.rs
        extrude_mesh.rs
        mod.rs
        vertex_extrusion.rs
```

and here is the source code:
`src/extrusion/mod.rs`

```rust
pub mod core;
pub mod infrastructure;
pub mod interface_adapters;
pub mod use_cases;

#[cfg(test)]
mod tests;
```

`src/extrusion/tests.rs`

```rust
#[cfg(test)]
mod tests {
    use crate::extrusion::{
        core::{hexahedral_mesh::QuadrilateralMesh, prismatic_mesh::TriangularMesh},
        infrastructure::logger::Logger,
        interface_adapters::extrusion_service::ExtrusionService,
        use_cases::extrude_mesh::ExtrudeMeshUseCase,
    };
    use crate::input_output::gmsh_parser::GmshParser;
    use crate::domain::mesh_entity::MeshEntity;

    #[test]
    fn test_extrude_rectangle_to_hexahedron() {
        let temp_file_path = "inputs/rectangle_quad.msh2";
        let result = GmshParser::from_gmsh_file(temp_file_path);
        assert!(result.is_ok());

        let mesh_2d = result.unwrap();
        let quad_mesh = QuadrilateralMesh::new(mesh_2d.get_vertices(), mesh_2d.get_cell_vertex_indices());

        let depth = 5.0;
        let layers = 3;
        let extruded_mesh = ExtrudeMeshUseCase::extrude_to_hexahedron(&quad_mesh, depth, layers);
        assert!(extruded_mesh.is_ok());
        let extruded_mesh = extruded_mesh.unwrap();

        // The expected number of vertices should be (nx + 1) * (ny + 1) * (layers + 1)
        let expected_vertices = 78 * (layers + 1); // 78 vertices in base 2D mesh
        let num_vertices = extruded_mesh
            .entities
            .read().expect("Failed to acquire read lock")
            .iter()
            .filter(|e| matches!(e, MeshEntity::Vertex(_)))
            .count();
        assert_eq!(num_vertices, expected_vertices, "Incorrect number of vertices in extruded hexahedron mesh");

        // The expected number of cells should be nx * ny * layers
        let expected_cells = 96 * layers; // 96 quadrilateral cells in base mesh
        let num_cells = extruded_mesh
            .entities
            .read().expect("Failed to acquire read lock")
            .iter()
            .filter(|e| matches!(e, MeshEntity::Cell(_)))
            .count();
        assert_eq!(num_cells, expected_cells, "Incorrect number of cells in extruded hexahedron mesh");
    }

    #[test]
    fn test_extrude_triangle_basin_to_prism() {
        let temp_file_path = "inputs/triangular_basin.msh2";
        let result = GmshParser::from_gmsh_file(temp_file_path);
        assert!(result.is_ok());

        let mesh_2d = result.unwrap();
        let tri_mesh = TriangularMesh::new(mesh_2d.get_vertices(), mesh_2d.get_cell_vertex_indices());

        let depth = 4.0;
        let layers = 2;
        let extruded_mesh = ExtrudeMeshUseCase::extrude_to_prism(&tri_mesh, depth, layers);
        assert!(extruded_mesh.is_ok());
        let extruded_mesh = extruded_mesh.unwrap();

        // The expected number of vertices should be num_vertices_2d * (layers + 1)
        let expected_vertices = 66 * (layers + 1); // 66 vertices in base 2D triangular mesh
        let num_vertices = extruded_mesh
            .entities
            .read().expect("Failed to acquire read lock")
            .iter()
            .filter(|e| matches!(e, MeshEntity::Vertex(_)))
            .count();
        assert_eq!(num_vertices, expected_vertices, "Incorrect number of vertices in extruded prismatic mesh");

        // The expected number of cells should be num_cells_2d * layers
        let expected_cells = 133 * layers; // 133 triangular cells in base mesh
        let num_cells = extruded_mesh
            .entities
            .read().expect("Failed to acquire read lock")
            .iter()
            .filter(|e| matches!(e, MeshEntity::Cell(_)))
            .count();
        assert_eq!(num_cells, expected_cells, "Incorrect number of cells in extruded prismatic mesh");
    }

    #[test]
    fn test_extrusion_service_for_hexahedron() {
        let temp_file_path = "inputs/rectangle_quad.msh2";
        let result = GmshParser::from_gmsh_file(temp_file_path);
        assert!(result.is_ok());

        let mesh_2d = result.unwrap();
        let quad_mesh = QuadrilateralMesh::new(mesh_2d.get_vertices(), mesh_2d.get_cell_vertex_indices());

        let depth = 3.0;
        let layers = 2;
        let extruded_mesh = ExtrusionService::extrude_mesh(&quad_mesh, depth, layers);
        assert!(extruded_mesh.is_ok());

        // Basic assertions about the 3D mesh structure
        let extruded_mesh = extruded_mesh.unwrap();
        assert!(extruded_mesh.count_entities(&MeshEntity::Vertex(0)) > 0);
        assert!(extruded_mesh.count_entities(&MeshEntity::Cell(0)) > 0);
    }

    #[test]
    fn test_logger_functionality() {
        let mut logger = Logger::new(None).expect("Failed to initialize logger");
        logger.info("Starting extrusion test for hexahedral mesh");
        logger.warn("Mesh contains irregular vertices");
        logger.error("Extrusion failed due to invalid cell data");

        // Check if the logger runs without error; manual check of output is recommended.
    }
}
```

`src/extrusion/core/extrudable_mesh.rs`

```rust
use super::{hexahedral_mesh::QuadrilateralMesh, prismatic_mesh::TriangularMesh};
use std::fmt::Debug;

/// The `ExtrudableMesh` trait defines the required methods for a 2D mesh that is capable of extrusion,
/// transforming it from a 2D to a 3D representation. This trait supports handling different mesh types,
/// specifically quadrilateral and triangular meshes, and provides methods for downcasting to specific types.
///
/// Implementations of this trait should ensure that the mesh is compatible with extrusion (e.g., only quads or triangles),
/// and provide vertices and cell connectivity for the mesh in a 3D extrusion context.
pub trait ExtrudableMesh: Debug {
    /// Checks if the mesh is valid for extrusion by ensuring all cells adhere to the expected type
    /// (e.g., all cells are quads or all are triangles).
    ///
    /// # Returns
    ///
    /// - `bool`: `true` if the mesh is valid for extrusion, `false` otherwise.
    fn is_valid_for_extrusion(&self) -> bool;

    /// Returns a list of vertices in the mesh, with each vertex formatted as a 3D coordinate.
    ///
    /// # Returns
    ///
    /// - `Vec<[f64; 3]>`: A vector of 3D coordinates representing the vertices of the 2D mesh.
    fn get_vertices(&self) -> Vec<[f64; 3]>;

    /// Returns the cell-to-vertex connectivity of the mesh, where each cell is represented by indices
    /// that refer to vertices in the `get_vertices` array.
    ///
    /// # Returns
    ///
    /// - `Vec<Vec<usize>>`: A vector of cells, each of which is a list of vertex indices.
    fn get_cells(&self) -> Vec<Vec<usize>>;

    /// Determines if this mesh is a quadrilateral mesh.
    ///
    /// # Returns
    ///
    /// - `bool`: `true` if the mesh is of type `QuadrilateralMesh`, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// assert!(some_mesh.is_quad_mesh());
    /// ```
    fn is_quad_mesh(&self) -> bool {
        self.as_any().is::<QuadrilateralMesh>()
    }

    /// Determines if this mesh is a triangular mesh.
    ///
    /// # Returns
    ///
    /// - `bool`: `true` if the mesh is of type `TriangularMesh`, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// assert!(some_mesh.is_tri_mesh());
    /// ```
    fn is_tri_mesh(&self) -> bool {
        self.as_any().is::<TriangularMesh>()
    }

    /// Attempts to cast this mesh to a `QuadrilateralMesh` reference.
    ///
    /// # Returns
    ///
    /// - `Option<&QuadrilateralMesh>`: Some reference if successful, `None` otherwise.
    fn as_quad(&self) -> Option<&QuadrilateralMesh> {
        self.as_any().downcast_ref::<QuadrilateralMesh>()
    }

    /// Attempts to cast this mesh to a `TriangularMesh` reference.
    ///
    /// # Returns
    ///
    /// - `Option<&TriangularMesh>`: Some reference if successful, `None` otherwise.
    fn as_tri(&self) -> Option<&TriangularMesh> {
        self.as_any().downcast_ref::<TriangularMesh>()
    }

    /// Provides a type-erased reference to the mesh to allow downcasting to a specific type.
    ///
    /// # Returns
    ///
    /// - `&dyn Any`: A reference to the mesh as an `Any` type.
    fn as_any(&self) -> &dyn std::any::Any;
}

#[cfg(test)]
mod tests {
    use super::ExtrudableMesh;
    use crate::extrusion::core::{hexahedral_mesh::QuadrilateralMesh, prismatic_mesh::TriangularMesh};

    #[test]
    /// Tests the `is_valid_for_extrusion` method for both quadrilateral and triangular meshes.
    /// Verifies that valid meshes return `true`, while invalid meshes return `false`.
    fn test_is_valid_for_extrusion() {
        // Valid quadrilateral mesh
        let quad_mesh = QuadrilateralMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            vec![vec![0, 1, 2, 3]],
        );
        assert!(quad_mesh.is_valid_for_extrusion(), "Valid quadrilateral mesh should return true");

        // Valid triangular mesh
        let tri_mesh = TriangularMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            vec![vec![0, 1, 2]],
        );
        assert!(tri_mesh.is_valid_for_extrusion(), "Valid triangular mesh should return true");
    }

    #[test]
    /// Tests the `get_vertices` method to ensure it returns the correct vertices for both quadrilateral and triangular meshes.
    fn test_get_vertices() {
        let quad_vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
        let quad_mesh = QuadrilateralMesh::new(quad_vertices.clone(), vec![vec![0, 1, 2, 3]]);
        assert_eq!(quad_mesh.get_vertices(), quad_vertices, "Quadrilateral vertices should match the input");

        let tri_vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];
        let tri_mesh = TriangularMesh::new(tri_vertices.clone(), vec![vec![0, 1, 2]]);
        assert_eq!(tri_mesh.get_vertices(), tri_vertices, "Triangular vertices should match the input");
    }

    #[test]
    /// Tests the `get_cells` method to ensure it returns the correct cell connectivity for both quadrilateral and triangular meshes.
    fn test_get_cells() {
        let quad_cells = vec![vec![0, 1, 2, 3]];
        let quad_mesh = QuadrilateralMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            quad_cells.clone(),
        );
        assert_eq!(quad_mesh.get_cells(), quad_cells, "Quadrilateral cells should match the input");

        let tri_cells = vec![vec![0, 1, 2]];
        let tri_mesh = TriangularMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            tri_cells.clone(),
        );
        assert_eq!(tri_mesh.get_cells(), tri_cells, "Triangular cells should match the input");
    }

    #[test]
    /// Tests `is_quad_mesh` and `is_tri_mesh` methods to verify correct type identification for quadrilateral and triangular meshes.
    fn test_mesh_type_identification() {
        let quad_mesh = QuadrilateralMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            vec![vec![0, 1, 2, 3]],
        );
        assert!(quad_mesh.is_quad_mesh(), "Quadrilateral mesh should identify as quad mesh");
        assert!(!quad_mesh.is_tri_mesh(), "Quadrilateral mesh should not identify as tri mesh");

        let tri_mesh = TriangularMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            vec![vec![0, 1, 2]],
        );
        assert!(tri_mesh.is_tri_mesh(), "Triangular mesh should identify as tri mesh");
        assert!(!tri_mesh.is_quad_mesh(), "Triangular mesh should not identify as quad mesh");
    }

    #[test]
    /// Tests `as_quad` and `as_tri` methods to verify proper downcasting to specific mesh types.
    fn test_mesh_downcasting() {
        let quad_mesh = QuadrilateralMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            vec![vec![0, 1, 2, 3]],
        );
        assert!(quad_mesh.as_quad().is_some(), "Downcast to QuadrilateralMesh should succeed");
        assert!(quad_mesh.as_tri().is_none(), "Downcast to TriangularMesh should fail");

        let tri_mesh = TriangularMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            vec![vec![0, 1, 2]],
        );
        assert!(tri_mesh.as_tri().is_some(), "Downcast to TriangularMesh should succeed");
        assert!(tri_mesh.as_quad().is_none(), "Downcast to QuadrilateralMesh should fail");
    }
}
```

`src/extrusion/core/hexahedral_mesh.rs`

```rust
use std::any::Any;
use crate::extrusion::core::extrudable_mesh::ExtrudableMesh;

/// The `QuadrilateralMesh` struct represents a 2D quadrilateral mesh, containing vertices
/// in 3D space and cells defined by indices of the vertices, each representing a quadrilateral.
///
/// This mesh struct is intended for extrusion into 3D hexahedral meshes.
///
/// # Example
///
/// ```
/// let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
/// let cells = vec![vec![0, 1, 2, 3]];
/// let quad_mesh = QuadrilateralMesh::new(vertices, cells);
/// assert!(quad_mesh.is_valid_for_extrusion());
/// 
/// ```
#[derive(Debug)]
pub struct QuadrilateralMesh {
    /// A list of vertices represented as `[f64; 3]` coordinates in 3D space.
    vertices: Vec<[f64; 3]>,
    /// A list of cells, where each cell is a `Vec<usize>` containing indices into the `vertices` array,
    /// representing the corners of each quadrilateral cell.
    cells: Vec<Vec<usize>>,
}

impl QuadrilateralMesh {
    /// Creates a new `QuadrilateralMesh` with the specified vertices and cells.
    ///
    /// # Parameters
    ///
    /// - `vertices`: A `Vec<[f64; 3]>` specifying the 3D coordinates of each vertex.
    /// - `cells`: A `Vec<Vec<usize>>` where each inner `Vec<usize>` contains 4 indices into `vertices`,
    ///   representing a quadrilateral cell.
    ///
    /// # Returns
    ///
    /// - `Self`: A new `QuadrilateralMesh` instance.
    ///
    /// # Example
    ///
    /// ```
    /// let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
    /// let cells = vec![vec![0, 1, 2, 3]];
    /// let quad_mesh = QuadrilateralMesh::new(vertices, cells);
    /// ```
    pub fn new(vertices: Vec<[f64; 3]>, cells: Vec<Vec<usize>>) -> Self {
        QuadrilateralMesh { vertices, cells }
    }
}

impl ExtrudableMesh for QuadrilateralMesh {
    /// Checks if the mesh is valid for extrusion.
    ///
    /// This method verifies that all cells in the mesh are quadrilateral (i.e., each cell
    /// contains exactly 4 vertices). If any cell does not contain 4 vertices, this function returns `false`.
    ///
    /// # Returns
    ///
    /// - `bool`: Returns `true` if all cells are quadrilateral; otherwise, `false`.
    ///
    /// # Example
    ///
    /// ```
    /// let quad_mesh = QuadrilateralMesh::new(...);
    /// assert!(quad_mesh.is_valid_for_extrusion());
    /// ```
    fn is_valid_for_extrusion(&self) -> bool {
        self.cells.iter().all(|cell| cell.len() == 4)
    }

    /// Returns a clone of the vertices in the mesh.
    ///
    /// # Returns
    ///
    /// - `Vec<[f64; 3]>`: A vector of 3D coordinates representing the mesh vertices.
    ///
    /// # Example
    ///
    /// ```
    /// let vertices = quad_mesh.get_vertices();
    /// ```
    fn get_vertices(&self) -> Vec<[f64; 3]> {
        self.vertices.clone()
    }

    /// Returns a clone of the cells in the mesh.
    ///
    /// # Returns
    ///
    /// - `Vec<Vec<usize>>`: A vector of cells, where each cell is defined by 4 vertex indices.
    ///
    /// # Example
    ///
    /// ```
    /// let cells = quad_mesh.get_cells();
    /// ```
    fn get_cells(&self) -> Vec<Vec<usize>> {
        self.cells.clone()
    }

    /// Provides a type-erased reference to the current object, allowing it to be used
    /// as a generic `ExtrudableMesh` object.
    ///
    /// # Returns
    ///
    /// - `&dyn Any`: A type-erased reference to the mesh.
    ///
    /// # Example
    ///
    /// ```
    /// let as_any = quad_mesh.as_any();
    /// ```
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::QuadrilateralMesh;
    use crate::extrusion::core::extrudable_mesh::ExtrudableMesh;

    #[test]
    /// Tests the creation of a `QuadrilateralMesh` instance.
    /// Verifies that the mesh initializes correctly with the provided vertices and cells.
    fn test_quadrilateral_mesh_creation() {
        let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
        let cells = vec![vec![0, 1, 2, 3]];
        
        let quad_mesh = QuadrilateralMesh::new(vertices.clone(), cells.clone());
        
        assert_eq!(quad_mesh.get_vertices(), vertices, "Vertices should match the input vertices");
        assert_eq!(quad_mesh.get_cells(), cells, "Cells should match the input cells");
    }

    #[test]
    /// Tests the `is_valid_for_extrusion` method of `QuadrilateralMesh`.
    /// Verifies that the mesh is valid only if all cells are quadrilateral.
    fn test_is_valid_for_extrusion() {
        let valid_mesh = QuadrilateralMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            vec![vec![0, 1, 2, 3]],
        );
        assert!(valid_mesh.is_valid_for_extrusion(), "Mesh with all quadrilateral cells should be valid");

        let invalid_mesh = QuadrilateralMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            vec![vec![0, 1, 2]],
        );
        assert!(!invalid_mesh.is_valid_for_extrusion(), "Mesh with non-quadrilateral cells should be invalid");
    }

    #[test]
    /// Tests the `get_vertices` method of `QuadrilateralMesh`.
    /// Verifies that `get_vertices` returns a clone of the original vertices.
    fn test_get_vertices() {
        let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
        let quad_mesh = QuadrilateralMesh::new(vertices.clone(), vec![vec![0, 1, 2, 3]]);
        
        assert_eq!(quad_mesh.get_vertices(), vertices, "Vertices should match the initialized vertices");
    }

    #[test]
    /// Tests the `get_cells` method of `QuadrilateralMesh`.
    /// Verifies that `get_cells` returns a clone of the original cells.
    fn test_get_cells() {
        let cells = vec![vec![0, 1, 2, 3]];
        let quad_mesh = QuadrilateralMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            cells.clone(),
        );
        
        assert_eq!(quad_mesh.get_cells(), cells, "Cells should match the initialized cells");
    }

    #[test]
    /// Tests the `as_any` method of `QuadrilateralMesh`.
    /// Verifies that the mesh can be treated as a `dyn Any` for type erasure.
    fn test_as_any() {
        let quad_mesh = QuadrilateralMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            vec![vec![0, 1, 2, 3]],
        );

        let as_any = quad_mesh.as_any();
        assert!(as_any.is::<QuadrilateralMesh>(), "as_any should identify the struct type correctly");
    }
}
```

`src/extrusion/core/mod.rs`

```rust
pub mod extrudable_mesh;
pub mod hexahedral_mesh;
pub mod prismatic_mesh;
```

`src/extrusion/core/prismatic_mesh.rs`

```rust
use std::any::Any;
use crate::extrusion::core::extrudable_mesh::ExtrudableMesh;

/// The `TriangularMesh` struct represents a 2D triangular mesh, consisting of vertices
/// in 3D space and cells defined by indices of the vertices, with each cell representing a triangle.
///
/// This mesh struct is designed to be extruded into 3D prismatic meshes.
///
/// # Example
///
/// ```
/// let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];
/// let cells = vec![vec![0, 1, 2]];
/// let tri_mesh = TriangularMesh::new(vertices, cells);
/// assert!(tri_mesh.is_valid_for_extrusion());
/// ```
#[derive(Debug)]
pub struct TriangularMesh {
    /// A list of vertices represented as `[f64; 3]` coordinates in 3D space.
    vertices: Vec<[f64; 3]>,
    /// A list of cells, where each cell is a `Vec<usize>` containing indices into the `vertices` array,
    /// representing the corners of each triangular cell.
    cells: Vec<Vec<usize>>,
}

impl TriangularMesh {
    /// Creates a new `TriangularMesh` with the specified vertices and cells.
    ///
    /// # Parameters
    ///
    /// - `vertices`: A `Vec<[f64; 3]>` specifying the 3D coordinates of each vertex.
    /// - `cells`: A `Vec<Vec<usize>>` where each inner `Vec<usize>` contains 3 indices into `vertices`,
    ///   representing a triangular cell.
    ///
    /// # Returns
    ///
    /// - `Self`: A new `TriangularMesh` instance.
    ///
    /// # Example
    ///
    /// ```
    /// let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];
    /// let cells = vec![vec![0, 1, 2]];
    /// let tri_mesh = TriangularMesh::new(vertices, cells);
    /// ```
    pub fn new(vertices: Vec<[f64; 3]>, cells: Vec<Vec<usize>>) -> Self {
        TriangularMesh { vertices, cells }
    }
}

impl ExtrudableMesh for TriangularMesh {
    /// Checks if the mesh is valid for extrusion.
    ///
    /// This method verifies that all cells in the mesh are triangular (i.e., each cell
    /// contains exactly 3 vertices). If any cell does not contain 3 vertices, this function returns `false`.
    ///
    /// # Returns
    ///
    /// - `bool`: Returns `true` if all cells are triangular; otherwise, `false`.
    ///
    /// # Example
    ///
    /// ```
    /// let tri_mesh = TriangularMesh::new(...);
    /// assert!(tri_mesh.is_valid_for_extrusion());
    /// ```
    fn is_valid_for_extrusion(&self) -> bool {
        self.cells.iter().all(|cell| cell.len() == 3)
    }

    /// Returns a clone of the vertices in the mesh.
    ///
    /// # Returns
    ///
    /// - `Vec<[f64; 3]>`: A vector of 3D coordinates representing the mesh vertices.
    ///
    /// # Example
    ///
    /// ```
    /// let vertices = tri_mesh.get_vertices();
    /// ```
    fn get_vertices(&self) -> Vec<[f64; 3]> {
        self.vertices.clone()
    }

    /// Returns a clone of the cells in the mesh.
    ///
    /// # Returns
    ///
    /// - `Vec<Vec<usize>>`: A vector of cells, where each cell is defined by 3 vertex indices.
    ///
    /// # Example
    ///
    /// ```
    /// let cells = tri_mesh.get_cells();
    /// ```
    fn get_cells(&self) -> Vec<Vec<usize>> {
        self.cells.clone()
    }

    /// Provides a type-erased reference to the current object, allowing it to be used
    /// as a generic `ExtrudableMesh` object.
    ///
    /// # Returns
    ///
    /// - `&dyn Any`: A type-erased reference to the mesh.
    ///
    /// # Example
    ///
    /// ```
    /// let as_any = tri_mesh.as_any();
    /// ```
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::TriangularMesh;
    use crate::extrusion::core::extrudable_mesh::ExtrudableMesh;

    #[test]
    /// Tests the creation of a `TriangularMesh` instance.
    /// Verifies that the mesh initializes correctly with the provided vertices and cells.
    fn test_triangular_mesh_creation() {
        let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];
        let cells = vec![vec![0, 1, 2]];

        let tri_mesh = TriangularMesh::new(vertices.clone(), cells.clone());

        assert_eq!(tri_mesh.get_vertices(), vertices, "Vertices should match the input vertices");
        assert_eq!(tri_mesh.get_cells(), cells, "Cells should match the input cells");
    }

    #[test]
    /// Tests the `is_valid_for_extrusion` method of `TriangularMesh`.
    /// Verifies that the mesh is valid only if all cells are triangular.
    fn test_is_valid_for_extrusion() {
        let valid_mesh = TriangularMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            vec![vec![0, 1, 2]],
        );
        assert!(valid_mesh.is_valid_for_extrusion(), "Mesh with all triangular cells should be valid");

        let invalid_mesh = TriangularMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            vec![vec![0, 1, 2, 3]],
        );
        assert!(!invalid_mesh.is_valid_for_extrusion(), "Mesh with non-triangular cells should be invalid");
    }

    #[test]
    /// Tests the `get_vertices` method of `TriangularMesh`.
    /// Verifies that `get_vertices` returns a clone of the original vertices.
    fn test_get_vertices() {
        let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];
        let tri_mesh = TriangularMesh::new(vertices.clone(), vec![vec![0, 1, 2]]);
        
        assert_eq!(tri_mesh.get_vertices(), vertices, "Vertices should match the initialized vertices");
    }

    #[test]
    /// Tests the `get_cells` method of `TriangularMesh`.
    /// Verifies that `get_cells` returns a clone of the original cells.
    fn test_get_cells() {
        let cells = vec![vec![0, 1, 2]];
        let tri_mesh = TriangularMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            cells.clone(),
        );
        
        assert_eq!(tri_mesh.get_cells(), cells, "Cells should match the initialized cells");
    }

    #[test]
    /// Tests the `as_any` method of `TriangularMesh`.
    /// Verifies that the mesh can be treated as a `dyn Any` for type erasure.
    fn test_as_any() {
        let tri_mesh = TriangularMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            vec![vec![0, 1, 2]],
        );

        let as_any = tri_mesh.as_any();
        assert!(as_any.is::<TriangularMesh>(), "as_any should identify the struct type correctly");
    }
}
```

`src/extrusion/infrastructure/logger.rs`

```rust
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::time::SystemTime;

/// Logger struct to handle logging with timestamps and levels.
pub struct Logger {
    output: Box<dyn Write + Send>,
}

impl Logger {
    /// Creates a new Logger that writes to the specified output.
    /// If a file path is provided, it logs to that file; otherwise, logs to stdout.
    pub fn new(file_path: Option<&str>) -> Result<Self, io::Error> {
        let output: Box<dyn Write + Send> = match file_path {
            Some(path) => Box::new(OpenOptions::new().create(true).append(true).open(path)?),
            None => Box::new(io::stdout()),
        };
        Ok(Logger { output })
    }

    /// Logs an info message with a timestamp.
    pub fn info(&mut self, message: &str) {
        self.log("INFO", message);
    }

    /// Logs a warning message with a timestamp.
    pub fn warn(&mut self, message: &str) {
        self.log("WARN", message);
    }

    /// Logs an error message with a timestamp.
    pub fn error(&mut self, message: &str) {
        self.log("ERROR", message);
    }

    /// Core logging function with a specified log level.
    fn log(&mut self, level: &str, message: &str) {
        let timestamp = match SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
            Ok(duration) => format!("{:?}", duration),
            Err(_) => "Unknown time".to_string(),
        };
        let formatted_message = format!("[{}] [{}] {}\n", timestamp, level, message);
        
        // Write the message to the output and flush
        if let Err(e) = self.output.write_all(formatted_message.as_bytes()) {
            eprintln!("Failed to write to log: {}", e);
        }
        if let Err(e) = self.output.flush() {
            eprintln!("Failed to flush log output: {}", e);
        }
    }
}
```

`src/extrusion/infrastructure/mesh_io.rs`

```rust
use crate::domain::{mesh::Mesh, mesh_entity::MeshEntity};
use crate::extrusion::core::{extrudable_mesh::ExtrudableMesh, hexahedral_mesh::QuadrilateralMesh, prismatic_mesh::TriangularMesh};
use crate::input_output::gmsh_parser::GmshParser;
use std::fs::File;
use std::io::Write;

/// `MeshIO` is responsible for handling input and output operations for mesh data, including
/// loading a 2D mesh from a file and saving an extruded 3D mesh.
pub struct MeshIO;

impl MeshIO {
    /// Loads a 2D mesh from a Gmsh file and returns it as an `ExtrudableMesh`.
    /// Detects the type of cells in the mesh (quadrilateral or triangular) and constructs
    /// an appropriate mesh structure based on the cell type.
    ///
    /// # Parameters
    ///
    /// - `file_path`: The path to the Gmsh file containing the 2D mesh data.
    ///
    /// # Returns
    ///
    /// - `Result<Box<dyn ExtrudableMesh>, String>`: Returns a boxed `ExtrudableMesh` trait
    ///   object, either a `QuadrilateralMesh` or `TriangularMesh`, or an error message if
    ///   the mesh type is unsupported or loading fails.
    ///
    /// # Errors
    ///
    /// - Returns an error if the Gmsh file cannot be read or if the mesh contains unsupported cell types.
    pub fn load_2d_mesh(file_path: &str) -> Result<Box<dyn ExtrudableMesh>, String> {
        let mesh = GmshParser::from_gmsh_file(file_path).map_err(|e| {
            format!("Failed to parse Gmsh file {}: {}", file_path, e.to_string())
        })?;
    
        let mut is_quad_mesh = true;
        let mut is_tri_mesh = true;
    
        for cell in mesh.get_cells() {
            let cell_vertex_count = mesh.get_cell_vertices(&cell).len();
            if cell_vertex_count == 4 {
                is_tri_mesh = false;
            } else if cell_vertex_count == 3 {
                is_quad_mesh = false;
            } else {
                return Err("Unsupported cell type: cells must be either quadrilateral or triangular.".to_string());
            }
        }
    
        // Instantiate the appropriate mesh type
        let result: Box<dyn ExtrudableMesh> = if is_quad_mesh {
            let quad_mesh = QuadrilateralMesh::new(
                mesh.get_vertices(),
                mesh.get_cell_vertex_indices(),
            );
            if quad_mesh.is_valid_for_extrusion() {
                Box::new(quad_mesh)
            } else {
                return Err("Invalid quadrilateral mesh structure".to_string());
            }
        } else if is_tri_mesh {
            let tri_mesh = TriangularMesh::new(
                mesh.get_vertices(),
                mesh.get_cell_vertex_indices(),
            );
            if tri_mesh.is_valid_for_extrusion() {
                Box::new(tri_mesh)
            } else {
                return Err("Invalid triangular mesh structure".to_string());
            }
        } else {
            return Err("Mesh must be exclusively quadrilateral or triangular.".to_string());
        };
    
        Ok(result)
    }

    /// Saves a 3D extruded mesh to a Gmsh-compatible file.
    ///
    /// # Parameters
    ///
    /// - `mesh`: A reference to the `Mesh` to save.
    /// - `file_path`: The path where the mesh will be saved.
    ///
    /// # Returns
    ///
    /// - `Result<(), String>`: Returns `Ok` if saving is successful, or an error message if
    ///   there is an I/O failure during the save process.
    ///
    /// # Errors
    ///
    /// - Returns an error if the file cannot be created or written to.
    pub fn save_3d_mesh(mesh: &Mesh, file_path: &str) -> Result<(), String> {
        let mut file = File::create(file_path).map_err(|e| {
            format!("Failed to create file {}: {}", file_path, e.to_string())
        })?;

        // Write vertices
        writeln!(file, "$Nodes").map_err(|e| e.to_string())?;
        writeln!(file, "{}", mesh.get_vertices().len()).map_err(|e| e.to_string())?;
        for (id, coords) in mesh.get_vertices().iter().enumerate() {
            writeln!(file, "{} {} {} {}", id + 1, coords[0], coords[1], coords[2])
                .map_err(|e| e.to_string())?;
        }
        writeln!(file, "$EndNodes").map_err(|e| e.to_string())?;

        // Write elements
        writeln!(file, "$Elements").map_err(|e| e.to_string())?;
        writeln!(file, "{}", mesh.get_cell_vertex_indices().len()).map_err(|e| e.to_string())?;
        for (id, vertices) in mesh.get_cell_vertex_indices().iter().enumerate() {
            writeln!(
                file,
                "{} 5 0 {}",
                id + 1,
                vertices.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" ")
            )
            .map_err(|e| format!("Failed to write element data: {}", e.to_string()))?;
        }
        writeln!(file, "$EndElements").map_err(|e| e.to_string())?;

        Ok(())
    }
}

impl Mesh {
    /// Retrieves all vertices in the mesh as a vector of `[f64; 3]` coordinates.
    pub fn get_vertices(&self) -> Vec<[f64; 3]> {
        self.entities
            .read()
            .expect("Failed to acquire read lock")
            .iter()
            .filter_map(|entity| match entity {
                MeshEntity::Vertex(id) => self.vertex_coordinates.get(id).cloned(),
                _ => None,
            })
            .collect()
    }

    /// Retrieves vertex indices (IDs) for all cells as a vector of `Vec<usize>`.
    pub fn get_cell_vertex_indices(&self) -> Vec<Vec<usize>> {
        self.entities
            .read()
            .expect("Failed to acquire read lock")
            .iter()
            .filter_map(|entity| match entity {
                MeshEntity::Cell(id) => Some(self.get_cell_vertex_ids(&MeshEntity::Cell(*id))),
                _ => None,
            })
            .collect()
    }

    /// Helper function to retrieve only vertex IDs for a cell.
    pub fn get_cell_vertex_ids(&self, cell: &MeshEntity) -> Vec<usize> {
        self.sieve
            .cone(cell)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|entity| match entity {
                MeshEntity::Vertex(vertex_id) => Some(vertex_id),
                _ => None,
            })
            .collect()
    }
}


#[cfg(test)]
mod tests {
    use super::MeshIO;
    use crate::domain::mesh::Mesh;
    use std::fs;

    #[test]
    fn test_load_quadrilateral_mesh() {
        let file_path = "inputs/rectangular_channel_quad.msh2";

        assert!(
            std::path::Path::new(file_path).exists(),
            "File not found: {}",
            file_path
        );

        let result = MeshIO::load_2d_mesh(file_path);

        match result {
            Ok(mesh) => {
                assert!(mesh.is_quad_mesh(), "Loaded mesh should be recognized as a quadrilateral mesh");
            },
            Err(e) => {
                eprintln!("Error loading mesh: {}", e);
                panic!("Expected successful loading of quadrilateral mesh");
            }
        }
    }

    #[test]
    /// Validates loading of a triangular mesh from a Gmsh file and its conversion to a `TriangularMesh`.
    fn test_load_triangular_mesh() {
        let file_path = "inputs/triangular_basin.msh2";
        let result = MeshIO::load_2d_mesh(file_path);

        assert!(result.is_ok(), "Expected successful loading of triangular mesh");
        let mesh = result.unwrap();
        assert!(mesh.is_tri_mesh(), "Loaded mesh should be recognized as a triangular mesh");
    }

    #[test]
    /// Tests saving of a 3D extruded mesh and verifies file creation and content.
    fn test_save_3d_mesh() {
        // Create a simple 3D mesh to save
        let mut mesh = Mesh::new();
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];

        for (id, vertex) in vertices.into_iter().enumerate() {
            mesh.set_vertex_coordinates(id, vertex);
        }

        let cell = crate::domain::mesh_entity::MeshEntity::Cell(0);
        mesh.add_entity(cell.clone());
        mesh.add_relationship(cell.clone(), crate::domain::mesh_entity::MeshEntity::Vertex(0));
        mesh.add_relationship(cell.clone(), crate::domain::mesh_entity::MeshEntity::Vertex(1));
        mesh.add_relationship(cell.clone(), crate::domain::mesh_entity::MeshEntity::Vertex(2));
        mesh.add_relationship(cell.clone(), crate::domain::mesh_entity::MeshEntity::Vertex(3));

        let file_path = "outputs/test_save_3d_mesh.msh";
        let result = MeshIO::save_3d_mesh(&mesh, file_path);

        assert!(result.is_ok(), "Expected successful saving of 3D mesh");
        
        // Check if file was created
        let file_exists = fs::metadata(file_path).is_ok();
        assert!(file_exists, "File should be created at specified path");

        // Cleanup
        fs::remove_file(file_path).expect("Failed to delete test output file");
    }

    #[test]
    /// Tests error handling when saving a mesh fails due to an invalid file path.
    fn test_save_3d_mesh_invalid_path() {
        let mesh = Mesh::new();
        let file_path = "/invalid_path/test_save_3d_mesh.msh";
        let result = MeshIO::save_3d_mesh(&mesh, file_path);

        assert!(result.is_err(), "Expected error when saving to invalid path");
    }
}
```

`src/extrusion/infrastructure/mod.rs`

```rust
pub mod mesh_io;
pub mod logger;
```

`src/extrusion/interface_adapters/extrusion_service.rs`

```rust
use crate::extrusion::core::extrudable_mesh::ExtrudableMesh;
use crate::extrusion::use_cases::extrude_mesh::ExtrudeMeshUseCase;
use crate::domain::mesh::Mesh;

/// `ExtrusionService` serves as the main interface for extruding a 2D mesh into a 3D mesh.
/// It supports both quadrilateral and triangular meshes, leveraging the mesh's type to determine
/// the appropriate extrusion method (hexahedral or prismatic).
pub struct ExtrusionService;

impl ExtrusionService {
    /// Extrudes a 2D mesh into a 3D mesh, determining the mesh type (quad or triangle) and
    /// extruding it accordingly.
    ///
    /// # Parameters
    ///
    /// - `mesh`: A reference to a 2D mesh that implements the `ExtrudableMesh` trait, indicating
    ///   the mesh supports extrusion operations.
    /// - `depth`: The extrusion depth, specifying the total height of the extruded 3D mesh.
    /// - `layers`: The number of layers into which the extrusion is divided.
    ///
    /// # Returns
    ///
    /// - `Result<Mesh, String>`: Returns `Ok` with the extruded 3D `Mesh` on success, or an
    ///   error message `String` if extrusion fails.
    ///
    /// # Errors
    ///
    /// - Returns an error if the mesh type is unsupported or the downcasting to a specific mesh type fails.
    ///
    /// # Example
    ///
    /// ```
    /// let quad_mesh = QuadrilateralMesh::new(...);
    /// let depth = 5.0;
    /// let layers = 3;
    /// let result = ExtrusionService::extrude_mesh(&quad_mesh, depth, layers);
    /// ```
    pub fn extrude_mesh(mesh: &dyn ExtrudableMesh, depth: f64, layers: usize) -> Result<Mesh, String> {
        if mesh.is_quad_mesh() {
            let quad_mesh = mesh.as_quad().ok_or("Failed to downcast to QuadrilateralMesh")?;
            ExtrudeMeshUseCase::extrude_to_hexahedron(quad_mesh, depth, layers)
        } else if mesh.is_tri_mesh() {
            let tri_mesh = mesh.as_tri().ok_or("Failed to downcast to TriangularMesh")?;
            ExtrudeMeshUseCase::extrude_to_prism(tri_mesh, depth, layers)
        } else {
            Err("Unsupported mesh type".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ExtrusionService;
    use crate::extrusion::core::{hexahedral_mesh::QuadrilateralMesh, prismatic_mesh::TriangularMesh};
    use crate::extrusion::core::extrudable_mesh::ExtrudableMesh;

    #[test]
    /// Validates the extrusion of a quadrilateral mesh into a hexahedral mesh.
    fn test_extrude_quad_mesh_to_hexahedron() {
        // Create a simple quadrilateral mesh
        let quad_mesh = QuadrilateralMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            vec![vec![0, 1, 2, 3]],
        );
        let depth = 5.0;
        let layers = 3;

        // Perform extrusion
        let extruded_result = ExtrusionService::extrude_mesh(&quad_mesh, depth, layers);

        // Check that extrusion is successful and returns a valid Mesh
        assert!(extruded_result.is_ok(), "Extrusion should succeed for quadrilateral mesh");
        let extruded_mesh = extruded_result.unwrap();
        assert!(extruded_mesh.count_entities(&crate::domain::mesh_entity::MeshEntity::Cell(0)) > 0, 
            "Extruded mesh should contain hexahedral cells");
    }

    #[test]
    /// Validates the extrusion of a triangular mesh into a prismatic mesh.
    fn test_extrude_tri_mesh_to_prism() {
        // Create a simple triangular mesh
        let tri_mesh = TriangularMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            vec![vec![0, 1, 2]],
        );
        let depth = 4.0;
        let layers = 2;

        // Perform extrusion
        let extruded_result = ExtrusionService::extrude_mesh(&tri_mesh, depth, layers);

        // Check that extrusion is successful and returns a valid Mesh
        assert!(extruded_result.is_ok(), "Extrusion should succeed for triangular mesh");
        let extruded_mesh = extruded_result.unwrap();
        assert!(extruded_mesh.count_entities(&crate::domain::mesh_entity::MeshEntity::Cell(0)) > 0, 
            "Extruded mesh should contain prismatic cells");
    }

    #[test]
    /// Tests that attempting to extrude an unsupported mesh type returns an error.
    fn test_unsupported_mesh_type() {
        #[derive(Debug)]
        struct UnsupportedMesh;
        impl ExtrudableMesh for UnsupportedMesh {
            fn is_valid_for_extrusion(&self) -> bool { false }
            fn get_vertices(&self) -> Vec<[f64; 3]> { vec![] }
            fn get_cells(&self) -> Vec<Vec<usize>> { vec![] }
            fn as_any(&self) -> &dyn std::any::Any { self }
        }

        let unsupported_mesh = UnsupportedMesh;
        let depth = 5.0;
        let layers = 3;

        // Attempt extrusion and expect an error
        let extruded_result = ExtrusionService::extrude_mesh(&unsupported_mesh, depth, layers);
        assert!(extruded_result.is_err(), "Extrusion should fail for unsupported mesh type");
        assert_eq!(extruded_result.unwrap_err(), "Unsupported mesh type", "Error message should indicate unsupported mesh type");
    }
}
```

`src/extrusion/interface_adapters/mod.rs`

```rust
pub mod extrusion_service;
```

`src/extrusion/use_cases/cell_extrusion.rs`

```rust
/// The `CellExtrusion` struct provides methods for extruding 2D cell structures (such as quadrilaterals and triangles)
/// into 3D volumetric cells (such as hexahedrons and prisms) across multiple layers.
///
/// This is a core part of mesh generation for 3D modeling, where cells from a 2D mesh are extended
/// in the z-direction to create volumetric cells.
///
/// # Example
///
/// ```
/// let quad_cells = vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]];
/// let layers = 3;
/// let extruded_cells = CellExtrusion::extrude_quadrilateral_cells(quad_cells, layers);
/// ```
pub struct CellExtrusion;

impl CellExtrusion {
    /// Extrudes a vector of quadrilateral cells into hexahedral cells across multiple layers.
    ///
    /// Each quadrilateral cell is defined by four vertices `[v0, v1, v2, v3]`, which represent
    /// the cell's vertices in a counter-clockwise or clockwise order. This method extrudes each
    /// cell along the z-axis to create a hexahedral cell with eight vertices.
    ///
    /// # Parameters
    ///
    /// - `cells`: A `Vec<Vec<usize>>` where each inner vector represents a quadrilateral cell by storing
    ///   four vertex indices.
    /// - `layers`: An `usize` representing the number of layers to extrude. Each cell will be extended
    ///   into `layers` hexahedral cells.
    ///
    /// # Returns
    ///
    /// A `Vec<Vec<usize>>` where each inner vector represents a hexahedral cell, defined by eight vertices.
    ///
    /// # Example
    ///
    /// ```
    /// let quad_cells = vec![vec![0, 1, 2, 3]];
    /// let layers = 2;
    /// let extruded_cells = CellExtrusion::extrude_quadrilateral_cells(quad_cells, layers);
    ///
    /// assert_eq!(extruded_cells.len(), 2); // Two layers of hexahedral cells
    /// ```
    pub fn extrude_quadrilateral_cells(cells: Vec<Vec<usize>>, layers: usize) -> Vec<Vec<usize>> {
        let mut extruded_cells = Vec::with_capacity(cells.len() * layers);

        for layer in 0..layers {
            let offset = layer * cells.len();
            let next_offset = (layer + 1) * cells.len();

            for cell in &cells {
                // Each quadrilateral cell [v0, v1, v2, v3] is extruded into a hexahedron with 8 vertices
                let hexahedron = vec![
                    offset + cell[0], offset + cell[1], offset + cell[2], offset + cell[3],
                    next_offset + cell[0], next_offset + cell[1], next_offset + cell[2], next_offset + cell[3],
                ];
                extruded_cells.push(hexahedron);
            }
        }

        extruded_cells
    }

    /// Extrudes a vector of triangular cells into prismatic cells across multiple layers.
    ///
    /// Each triangular cell is defined by three vertices `[v0, v1, v2]`, which represent
    /// the cell's vertices in a counter-clockwise or clockwise order. This method extrudes each
    /// cell along the z-axis to create a prismatic cell with six vertices.
    ///
    /// # Parameters
    ///
    /// - `cells`: A `Vec<Vec<usize>>` where each inner vector represents a triangular cell by storing
    ///   three vertex indices.
    /// - `layers`: An `usize` representing the number of layers to extrude. Each cell will be extended
    ///   into `layers` prismatic cells.
    ///
    /// # Returns
    ///
    /// A `Vec<Vec<usize>>` where each inner vector represents a prismatic cell, defined by six vertices.
    ///
    /// # Example
    ///
    /// ```
    /// let tri_cells = vec![vec![0, 1, 2]];
    /// let layers = 2;
    /// let extruded_cells = CellExtrusion::extrude_triangular_cells(tri_cells, layers);
    ///
    /// assert_eq!(extruded_cells.len(), 2); // Two layers of prismatic cells
    /// ```
    pub fn extrude_triangular_cells(cells: Vec<Vec<usize>>, layers: usize) -> Vec<Vec<usize>> {
        let mut extruded_cells = Vec::with_capacity(cells.len() * layers);

        for layer in 0..layers {
            let offset = layer * cells.len();
            let next_offset = (layer + 1) * cells.len();

            for cell in &cells {
                // Each triangular cell [v0, v1, v2] is extruded into a prism with 6 vertices
                let prism = vec![
                    offset + cell[0], offset + cell[1], offset + cell[2],
                    next_offset + cell[0], next_offset + cell[1], next_offset + cell[2],
                ];
                extruded_cells.push(prism);
            }
        }

        extruded_cells
    }
}

#[cfg(test)]
mod tests {
    use super::CellExtrusion;

    #[test]
    /// Test extrusion of quadrilateral cells across multiple layers.
    /// This test verifies that each quadrilateral cell is correctly transformed into a hexahedral cell
    /// and that the expected number of extruded cells are generated.
    fn test_extrude_quadrilateral_cells() {
        let quad_cells = vec![vec![0, 1, 2, 3]];
        let layers = 3;

        let extruded_cells = CellExtrusion::extrude_quadrilateral_cells(quad_cells.clone(), layers);

        // Expect 3 hexahedral layers for each quadrilateral cell
        assert_eq!(extruded_cells.len(), quad_cells.len() * layers);

        // Check the structure of the extruded hexahedral cells
        for layer in 0..layers {
            let offset = layer * quad_cells.len();
            let next_offset = (layer + 1) * quad_cells.len();

            for (i, cell) in quad_cells.iter().enumerate() {
                let hexahedron = extruded_cells[layer * quad_cells.len() + i].clone();
                assert_eq!(hexahedron, vec![
                    offset + cell[0], offset + cell[1], offset + cell[2], offset + cell[3],
                    next_offset + cell[0], next_offset + cell[1], next_offset + cell[2], next_offset + cell[3],
                ]);
            }
        }
    }

    #[test]
    /// Test extrusion of triangular cells across multiple layers.
    /// This test checks that each triangular cell is transformed into a prismatic cell
    /// and that the correct number of extruded cells are produced.
    fn test_extrude_triangular_cells() {
        let tri_cells = vec![vec![0, 1, 2]];
        let layers = 2;

        let extruded_cells = CellExtrusion::extrude_triangular_cells(tri_cells.clone(), layers);

        // Expect 2 prismatic layers for each triangular cell
        assert_eq!(extruded_cells.len(), tri_cells.len() * layers);

        // Check the structure of the extruded prismatic cells
        for layer in 0..layers {
            let offset = layer * tri_cells.len();
            let next_offset = (layer + 1) * tri_cells.len();

            for (i, cell) in tri_cells.iter().enumerate() {
                let prism = extruded_cells[layer * tri_cells.len() + i].clone();
                assert_eq!(prism, vec![
                    offset + cell[0], offset + cell[1], offset + cell[2],
                    next_offset + cell[0], next_offset + cell[1], next_offset + cell[2],
                ]);
            }
        }
    }

    #[test]
    /// Test extrusion with a single layer for quadrilateral cells.
    /// This ensures that the function handles single-layer extrusion correctly.
    fn test_single_layer_quadrilateral_extrusion() {
        let quad_cells = vec![vec![0, 1, 2, 3]];
        let layers = 1;

        let extruded_cells = CellExtrusion::extrude_quadrilateral_cells(quad_cells.clone(), layers);

        assert_eq!(extruded_cells.len(), quad_cells.len()); // Only one layer should be extruded

        // Verify that the single layer's z-offset behaves as expected
        let hexahedron = extruded_cells[0].clone();
        assert_eq!(hexahedron, vec![
            0, 1, 2, 3,
            quad_cells.len() + 0, quad_cells.len() + 1, quad_cells.len() + 2, quad_cells.len() + 3,
        ]);
    }

    #[test]
    /// Test extrusion with zero layers to ensure it gracefully handles edge cases without error.
    fn test_zero_layers_extrusion() {
        let quad_cells = vec![vec![0, 1, 2, 3]];
        let layers = 0;

        let extruded_cells = CellExtrusion::extrude_quadrilateral_cells(quad_cells.clone(), layers);
        assert!(extruded_cells.is_empty(), "Extruded cells should be empty when layers is zero.");
    }
}
```

`src/extrusion/use_cases/extrude_mesh.rs`

```rust
use crate::extrusion::core::{hexahedral_mesh::QuadrilateralMesh, prismatic_mesh::TriangularMesh};
use crate::extrusion::core::extrudable_mesh::ExtrudableMesh;
use crate::extrusion::use_cases::{vertex_extrusion::VertexExtrusion, cell_extrusion::CellExtrusion};
use crate::domain::mesh::Mesh;

/// The `ExtrudeMeshUseCase` struct provides methods for extruding 2D meshes (either quadrilateral or triangular)
/// into 3D volumetric meshes (hexahedrons or prisms) based on a given extrusion depth and layer count.
///
/// This struct builds upon lower-level extrusion operations for vertices and cells, assembling a fully extruded
/// 3D mesh by extruding the vertices and connecting them in new 3D cells.
///
/// # Example
///
/// ```
/// let mesh = QuadrilateralMesh::new(...);
/// let depth = 5.0;
/// let layers = 3;
/// let extruded_mesh = ExtrudeMeshUseCase::extrude_to_hexahedron(&mesh, depth, layers);
/// ```
pub struct ExtrudeMeshUseCase;

impl ExtrudeMeshUseCase {
    /// Extrudes a quadrilateral 2D mesh into a 3D mesh with hexahedral cells.
    ///
    /// This method first extrudes the vertices to create multiple layers, then extrudes each quadrilateral cell
    /// into a hexahedral cell for each layer. Finally, it assembles the extruded vertices and cells into a
    /// `Mesh` structure, representing the final 3D mesh.
    ///
    /// # Parameters
    ///
    /// - `mesh`: A reference to a `QuadrilateralMesh`, which is the 2D quadrilateral mesh to be extruded.
    /// - `depth`: A `f64` specifying the total depth of extrusion along the z-axis.
    /// - `layers`: An `usize` indicating the number of layers to extrude.
    ///
    /// # Returns
    ///
    /// - `Result<Mesh, String>`: Returns `Ok(Mesh)` with the fully extruded mesh if successful,
    ///   or an `Err(String)` if the mesh is invalid for extrusion.
    ///
    /// # Example
    ///
    /// ```
    /// let quad_mesh = QuadrilateralMesh::new(...);
    /// let result = ExtrudeMeshUseCase::extrude_to_hexahedron(&quad_mesh, 10.0, 3);
    /// assert!(result.is_ok());
    /// ```
    pub fn extrude_to_hexahedron(mesh: &QuadrilateralMesh, depth: f64, layers: usize) -> Result<Mesh, String> {
        if !mesh.is_valid_for_extrusion() {
            return Err("Invalid mesh: Expected a quadrilateral mesh".to_string());
        }

        // Extrude vertices
        let extruded_vertices = VertexExtrusion::extrude_vertices(mesh.get_vertices(), depth, layers);

        // Extrude quadrilateral cells to hexahedrons
        let extruded_cells = CellExtrusion::extrude_quadrilateral_cells(mesh.get_cells(), layers);

        // Build the final Mesh
        let mut extruded_mesh = Mesh::new();
        for (id, vertex) in extruded_vertices.into_iter().enumerate() {
            extruded_mesh.set_vertex_coordinates(id, vertex);
        }

        for (cell_id, vertices) in extruded_cells.into_iter().enumerate() {
            let cell = crate::domain::mesh_entity::MeshEntity::Cell(cell_id);
            extruded_mesh.add_entity(cell.clone());
            for vertex in vertices {
                extruded_mesh.add_relationship(cell.clone(), crate::domain::mesh_entity::MeshEntity::Vertex(vertex));
            }
        }

        Ok(extruded_mesh)
    }

    /// Extrudes a triangular 2D mesh into a 3D mesh with prismatic cells.
    ///
    /// This method first extrudes the vertices to create multiple layers, then extrudes each triangular cell
    /// into a prismatic cell for each layer. Finally, it assembles the extruded vertices and cells into a
    /// `Mesh` structure, representing the final 3D mesh.
    ///
    /// # Parameters
    ///
    /// - `mesh`: A reference to a `TriangularMesh`, which is the 2D triangular mesh to be extruded.
    /// - `depth`: A `f64` specifying the total depth of extrusion along the z-axis.
    /// - `layers`: An `usize` indicating the number of layers to extrude.
    ///
    /// # Returns
    ///
    /// - `Result<Mesh, String>`: Returns `Ok(Mesh)` with the fully extruded mesh if successful,
    ///   or an `Err(String)` if the mesh is invalid for extrusion.
    ///
    /// # Example
    ///
    /// ```
    /// let tri_mesh = TriangularMesh::new(...);
    /// let result = ExtrudeMeshUseCase::extrude_to_prism(&tri_mesh, 5.0, 2);
    /// assert!(result.is_ok());
    /// ```
    pub fn extrude_to_prism(mesh: &TriangularMesh, depth: f64, layers: usize) -> Result<Mesh, String> {
        if !mesh.is_valid_for_extrusion() {
            return Err("Invalid mesh: Expected a triangular mesh".to_string());
        }

        // Extrude vertices
        let extruded_vertices = VertexExtrusion::extrude_vertices(mesh.get_vertices(), depth, layers);

        // Extrude triangular cells to prisms
        let extruded_cells = CellExtrusion::extrude_triangular_cells(mesh.get_cells(), layers);

        // Build the final Mesh
        let mut extruded_mesh = Mesh::new();
        for (id, vertex) in extruded_vertices.into_iter().enumerate() {
            extruded_mesh.set_vertex_coordinates(id, vertex);
        }

        for (cell_id, vertices) in extruded_cells.into_iter().enumerate() {
            let cell = crate::domain::mesh_entity::MeshEntity::Cell(cell_id);
            extruded_mesh.add_entity(cell.clone());
            for vertex in vertices {
                extruded_mesh.add_relationship(cell.clone(), crate::domain::mesh_entity::MeshEntity::Vertex(vertex));
            }
        }

        Ok(extruded_mesh)
    }
}

#[cfg(test)]
mod tests {
    use super::ExtrudeMeshUseCase;
    use crate::extrusion::core::{hexahedral_mesh::QuadrilateralMesh, prismatic_mesh::TriangularMesh};

    #[test]
    /// Test extruding a quadrilateral mesh to a hexahedral mesh.
    /// This test checks that the extruded mesh contains the correct number of vertices and cells.
    fn test_extrude_to_hexahedron() {
        let quad_mesh = QuadrilateralMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            vec![vec![0, 1, 2, 3]],
        );
        let depth = 3.0;
        let layers = 2;

        let result = ExtrudeMeshUseCase::extrude_to_hexahedron(&quad_mesh, depth, layers);
        assert!(result.is_ok(), "Extrusion should succeed for a valid quadrilateral mesh");

        let extruded_mesh = result.unwrap();
        assert_eq!(extruded_mesh.count_entities(&crate::domain::mesh_entity::MeshEntity::Vertex(0)), 4 * (layers + 1));
        assert_eq!(extruded_mesh.count_entities(&crate::domain::mesh_entity::MeshEntity::Cell(0)), layers);
    }

    #[test]
    /// Test extruding a triangular mesh to a prismatic mesh.
    /// This test checks that the extruded mesh contains the expected number of vertices and cells.
    fn test_extrude_to_prism() {
        let tri_mesh = TriangularMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            vec![vec![0, 1, 2]],
        );
        let depth = 2.0;
        let layers = 3;

        let result = ExtrudeMeshUseCase::extrude_to_prism(&tri_mesh, depth, layers);
        assert!(result.is_ok(), "Extrusion should succeed for a valid triangular mesh");

        let extruded_mesh = result.unwrap();
        assert_eq!(extruded_mesh.count_entities(&crate::domain::mesh_entity::MeshEntity::Vertex(0)), 3 * (layers + 1));
        assert_eq!(extruded_mesh.count_entities(&crate::domain::mesh_entity::MeshEntity::Cell(0)), layers);
    }
}
```

`src/extrusion/use_cases/mod.rs`

```rust
pub mod extrude_mesh;
pub mod vertex_extrusion;
pub mod cell_extrusion;
```

`src/extrusion/use_cases/vertex_extrusion.rs`

```rust
/// The `VertexExtrusion` struct provides methods for extruding a set of vertices along the z-axis.
/// This extrusion process is commonly used in mesh generation for three-dimensional models, where
/// a 2D base layer is extended in the z-direction to create a volumetric representation.
///
/// # Example
///
/// ```
/// let vertices = vec![[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]];
/// let depth = 10.0;
/// let layers = 5;
/// let extruded = VertexExtrusion::extrude_vertices(vertices, depth, layers);
/// ```
pub struct VertexExtrusion;

impl VertexExtrusion {
    /// Extrudes vertices along the z-axis, creating multiple layers of vertices based on
    /// the specified `depth` and number of `layers`.
    ///
    /// This function takes a set of base vertices defined in the XY plane (z = 0) and extrudes them
    /// along the z-axis to generate additional layers at regularly spaced intervals, forming a 
    /// three-dimensional structure.
    ///
    /// # Parameters
    ///
    /// - `vertices`: A `Vec<[f64; 3]>` representing the base vertices, each with an initial z-coordinate.
    /// - `depth`: A `f64` representing the total depth of the extrusion in the z-direction.
    /// - `layers`: An `usize` specifying the number of layers to generate. The function divides
    ///    the depth by this number to determine the z-coordinate increment (`dz`) between each layer.
    ///
    /// # Returns
    ///
    /// A `Vec<[f64; 3]>` containing the extruded vertices. The z-coordinate of each new layer
    /// increases by `dz` until reaching `depth`, thus forming layers from z = 0 to z = `depth`.
    ///
    /// # Panics
    ///
    /// This function does not panic as long as `layers > 0` (to avoid division by zero). If `layers`
    /// is zero, the caller should handle the case to prevent an undefined extrusion.
    ///
    /// # Example
    ///
    /// ```
    /// let vertices = vec![[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]];
    /// let depth = 10.0;
    /// let layers = 2;
    /// let extruded_vertices = VertexExtrusion::extrude_vertices(vertices, depth, layers);
    ///
    /// assert_eq!(extruded_vertices.len(), 6); // 3 layers x 2 vertices per layer
    /// ```
    pub fn extrude_vertices(vertices: Vec<[f64; 3]>, depth: f64, layers: usize) -> Vec<[f64; 3]> {
        let dz = depth / layers as f64;
        let mut extruded_vertices = Vec::with_capacity(vertices.len() * (layers + 1));

        for layer in 0..=layers {
            let z = dz * layer as f64;
            for vertex in &vertices {
                extruded_vertices.push([vertex[0], vertex[1], z]);
            }
        }

        extruded_vertices
    }
}

#[cfg(test)]
mod tests {
    use super::VertexExtrusion;

    #[test]
    /// Test that verifies the correct number of extruded vertices are generated for the
    /// specified layers, and that the z-coordinates increment correctly in each layer.
    fn test_extrusion_with_multiple_layers() {
        // Define a small set of vertices in the XY plane (z=0)
        let base_vertices = vec![[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]];
        let depth = 10.0;
        let layers = 5;

        // Perform the extrusion
        let extruded_vertices = VertexExtrusion::extrude_vertices(base_vertices.clone(), depth, layers);

        // The expected number of vertices is the original vertices count times the number of layers plus one
        assert_eq!(extruded_vertices.len(), base_vertices.len() * (layers + 1));

        // Check that each layer has the correct z-coordinate incrementally
        let dz = depth / layers as f64;
        for (i, z) in (0..=layers).map(|layer| dz * layer as f64).enumerate() {
            for j in 0..base_vertices.len() {
                let vertex_index = i * base_vertices.len() + j;
                let extruded_vertex = extruded_vertices[vertex_index];
                assert_eq!(extruded_vertex[0], base_vertices[j][0]);
                assert_eq!(extruded_vertex[1], base_vertices[j][1]);
                assert!((extruded_vertex[2] - z).abs() < 1e-6, "Incorrect z-coordinate in layer");
            }
        }
    }

    #[test]
    /// Test that verifies the function correctly extrudes vertices for a single layer,
    /// producing two sets of vertices: one at the base layer (z=0) and one at z=depth.
    fn test_extrusion_with_one_layer() {
        let base_vertices = vec![[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]];
        let depth = 5.0;
        let layers = 1;

        let extruded_vertices = VertexExtrusion::extrude_vertices(base_vertices.clone(), depth, layers);

        assert_eq!(extruded_vertices.len(), base_vertices.len() * 2); // Two layers

        // Verify z-coordinates for each layer
        for (i, z) in [0.0, depth].iter().enumerate() {
            for j in 0..base_vertices.len() {
                let vertex_index = i * base_vertices.len() + j;
                let extruded_vertex = extruded_vertices[vertex_index];
                assert_eq!(extruded_vertex[0], base_vertices[j][0]);
                assert_eq!(extruded_vertex[1], base_vertices[j][1]);
                assert!((extruded_vertex[2] - *z).abs() < 1e-6, "Incorrect z-coordinate for one layer extrusion");
            }
        }
    }

    #[test]
    /// Test that verifies the extrusion with multiple vertices and zero depth,
    /// resulting in no change along the z-axis across all layers.
    fn test_extrusion_with_zero_depth() {
        let base_vertices = vec![[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]];
        let depth = 0.0;
        let layers = 3;

        let extruded_vertices = VertexExtrusion::extrude_vertices(base_vertices.clone(), depth, layers);

        assert_eq!(extruded_vertices.len(), base_vertices.len() * (layers + 1));

        // Check that all extruded vertices have a z-coordinate of 0.0
        for extruded_vertex in extruded_vertices {
            assert_eq!(extruded_vertex[2], 0.0, "Extrusion with zero depth should have z=0 for all vertices");
        }
    }

    #[test]
    /// Test that verifies that an empty vertex list returns an empty extruded vertex list,
    /// ensuring no extraneous vertices are created when no input is given.
    fn test_extrusion_with_empty_vertices() {
        let base_vertices: Vec<[f64; 3]> = vec![];
        let depth = 5.0;
        let layers = 3;

        let extruded_vertices = VertexExtrusion::extrude_vertices(base_vertices.clone(), depth, layers);

        assert!(extruded_vertices.is_empty(), "Extrusion with empty vertices should result in empty output");
    }

    #[test]
    /// Test that checks the precision of extruded z-coordinates to ensure they are calculated
    /// correctly for non-integer values of `depth` and `layers`.
    fn test_extrusion_with_decimal_depth() {
        let base_vertices = vec![[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]];
        let depth = 3.75;
        let layers = 3;

        let extruded_vertices = VertexExtrusion::extrude_vertices(base_vertices.clone(), depth, layers);

        assert_eq!(extruded_vertices.len(), base_vertices.len() * (layers + 1));

        // Verify that the z-coordinates increment by depth/layers, with precision for decimal depth
        let dz = depth / layers as f64;
        for (i, z) in (0..=layers).map(|layer| dz * layer as f64).enumerate() {
            for j in 0..base_vertices.len() {
                let vertex_index = i * base_vertices.len() + j;
                let extruded_vertex = extruded_vertices[vertex_index];
                assert_eq!(extruded_vertex[0], base_vertices[j][0]);
                assert_eq!(extruded_vertex[1], base_vertices[j][1]);
                assert!((extruded_vertex[2] - z).abs() < 1e-6, "Incorrect z-coordinate with decimal depth");
            }
        }
    }
}
```