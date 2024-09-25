# HYDRA Contribution Policy

We welcome contributions from the community to help improve and expand the HYDRA project! To ensure a smooth and collaborative process, please follow the guidelines below when contributing to the project.

## How to Contribute

### 1. Fork the Repository
- Start by forking the HYDRA repository to your GitHub account.
- Clone the forked repository to your local machine:
  ```bash
  git clone https://github.com/your-username/HYDRA.git
  cd HYDRA
  ```

### 2. Create a Feature Branch
- Create a new branch for your contribution, named descriptively after the feature or bug fix you are working on:
  ```bash
  git checkout -b feature/my-new-feature
  ```

### 3. Make Your Changes
- Implement your feature or bug fix. Follow these best practices:
  - **Modularity**: Ensure that your contribution is modular and does not break existing functionality.
  - **Documentation**: If applicable, update or add new documentation (in the `docs/` folder or relevant source files).
  - **Code Style**: Follow Rust's standard coding conventions and maintain consistency with the existing codebase.

### 4. Add Tests
- Include tests for your changes, especially for new features or bug fixes. Tests help ensure that the contribution works as expected and doesn’t introduce regressions.
  - Place your tests in the appropriate module within the `src/` directory.
  - Run the tests to make sure everything works:
    ```bash
    cargo test
    ```

### 5. Commit Your Changes
- Commit your changes with a descriptive commit message:
  ```bash
  git add .
  git commit -m "Add feature: description of the feature"
  ```

### 6. Push Your Branch
- Push your feature branch to your forked repository:
  ```bash
  git push origin feature/my-new-feature
  ```

### 7. Open a Pull Request
- Once your branch is pushed, open a pull request (PR) to the main HYDRA repository:
  - Provide a clear title and description of the changes you made.
  - Mention any relevant issues that the PR addresses.
  - Explain any important design decisions or trade-offs.

### 8. Code Review and Collaboration
- Your PR will undergo a code review by the maintainers. Be prepared to discuss your changes and make revisions if necessary.
- Address any feedback promptly and update your PR until it’s ready for merging.

## Code of Conduct

We expect all contributors to follow our **Code of Conduct**, which promotes a positive, respectful, and inclusive community. Please review the Github code of conduct before contributing.

## Licensing

By contributing to HYDRA, you agree that your contributions will be licensed under the **MIT License**, in alignment with the project's overall licensing terms.

---

Thank you for your interest in contributing to HYDRA! Together, we can build a robust, efficient, and scalable tool for geophysical fluid dynamics simulations. If you have any questions, feel free to open an issue or join our community discussions.