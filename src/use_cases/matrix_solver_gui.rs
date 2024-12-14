use iced::{
    Application, Button, Column, Command, Element, Settings, Text, TextInput,
    executor, widget::{button, column, text, text_input},
};

#[derive(Default)]
struct GUI {
    file_input: String,
    solver_selected: String,
    result: Option<String>,
    input_state: text_input::State,
    load_button: button::State,
    solve_button: button::State,
}

#[derive(Debug, Clone)]
enum Message {
    FileChanged(String),
    SolverSelected(String),
    LoadFile,
    Solve,
    SolveComplete(Result<String, String>),
}

impl Application for GUI {
    type Executor = executor::Default;
    type Message = Message;
    type Flags = ();

    fn new(_: ()) -> (Self, Command<Self::Message>) {
        (Self::default(), Command::none())
    }

    fn title(&self) -> String {
        String::from("Hydra Solver GUI")
    }

    fn update(&mut self, message: Self::Message) -> Command<Self::Message> {
        match message {
            Message::FileChanged(path) => {
                self.file_input = path;
                Command::none()
            }
            Message::SolverSelected(solver) => {
                self.solver_selected = solver;
                Command::none()
            }
            Message::LoadFile => {
                // Logic for loading file
                Command::none()
            }
            Message::Solve => {
                let file_path = self.file_input.clone();
                Command::perform(
                    async move {
                        // Simulate solving the system
                        let mock_result = format!("Mock Solve Completed! Loaded: {}", file_path);
                        Ok(mock_result) as Result<_, String>
                    },
                    Message::SolveComplete,
                )
            }
            Message::SolveComplete(result) => {
                self.result = Some(result.unwrap_or_else(|e| e));
                Command::none()
            }
        }
    }

    fn view(&mut self) -> Element<Self::Message> {
        let file_input = text_input(
            "MatrixMarket File Path...",
            &self.file_input,
            Message::FileChanged,
        )
        .padding(10)
        .width(iced::Length::Fill);

        let load_button = button("Load").on_press(Message::LoadFile);
        let solve_button = button("Solve").on_press(Message::Solve);

        let result_text = text(self.result.clone().unwrap_or_else(|| String::from("No results yet")));

        column()
            .push(file_input)
            .push(load_button)
            .push(solve_button)
            .push(result_text)
            .spacing(20)
            .into()
    }

    type Theme;

    fn theme(&self) -> Self::Theme {
        Self::Theme::default()
    }

    fn style(&self) -> <Self::Theme as iced::application::StyleSheet>::Style {
        <Self::Theme as iced::application::StyleSheet>::Style::default()
    }

    fn subscription(&self) -> iced::Subscription<Self::Message> {
        iced::Subscription::none()
    }

    fn scale_factor(&self) -> f64 {
        1.0
    }

    fn run(settings: Settings<Self::Flags>) -> iced::Result
    where
        Self: 'static,
    {
        #[allow(clippy::needless_update)]
        let renderer_settings = iced::renderer::Settings {
            default_font: settings.default_font,
            default_text_size: settings.default_text_size,
            text_multithreading: settings.text_multithreading,
            antialiasing: if settings.antialiasing {
                Some(iced::renderer::settings::Antialiasing::MSAAx4)
            } else {
                None
            },
            ..iced::renderer::Settings::from_env()
        };

        Ok(iced::runtime::application::run::<
            Instance<Self>,
            Self::Executor,
            iced::renderer::window::Compositor<Self::Theme>,
        >(settings.into(), renderer_settings)?)
    }
}

pub fn main() -> iced::Result {
    GUI::run(Settings::default())
}


#[cfg(test)]
mod gui_tests {
    use super::*; use crate::interface_adapters::system_solver::SystemSolver;
    // Import the GUI and related modules
    use crate::solver::gmres::GMRES; use crate::solver::ksp::SolverResult;
    // Solver modules
    use crate::solver::preconditioner::PreconditionerFactory; // Preconditioner factories

    /// Mock SolverResult for testing purposes
    fn mock_solver_result() -> SolverResult {
        SolverResult {
            converged: true,
            iterations: 10,
            residual_norm: 1e-6,
        }
    }

    /// Simulates a user loading a MatrixMarket file
    #[test]
    fn test_gui_load_file() {
        let mut gui = GUI::default();
        let mock_file_path = "mock_matrix.mtx";

        // Simulate file input
        gui.update(Message::FileChanged(mock_file_path.to_string()));
        assert_eq!(gui.file_input, mock_file_path);

        // Simulate load button press
        gui.update(Message::LoadFile);
        // Add assertions if specific state changes are expected
    }

    /// Simulates the solve button workflow
    #[test]
    fn test_gui_solve_system() {
        let mut gui = GUI::default();
        let mock_file_path = "mock_matrix.mtx";
        gui.update(Message::FileChanged(mock_file_path.to_string()));

        // Simulate Solve button press
        gui.update(Message::Solve);

        // Verify that the solve process begins and GUI state changes as expected
        assert_eq!(gui.file_input, mock_file_path);

        // Simulate solve completion
        let mock_result = mock_solver_result();
        gui.update(Message::SolveComplete(Ok(format!(
            "Converged in {} iterations with residual norm {}",
            mock_result.iterations, mock_result.residual_norm
        ))));

        // Verify result display
        assert!(gui.result.is_some());
        if let Some(result) = &gui.result {
            assert!(result.contains("Converged"));
            assert!(result.contains("iterations"));
        }
    }

    /// Validates solver integration with the GUI using a mock MatrixMarket file
    #[test]
    fn test_solver_integration_with_gui() {
        let mock_matrix_file = "mock_matrix.mtx";

        // Create GMRES solver instance
        let gmres_solver = GMRES::new(100, 1e-6, 500);

        // Use a Jacobi preconditioner
        let preconditioner = Some(Box::new(PreconditionerFactory::create_jacobi) as Box<_>);

        // Simulate calling the solver via GUI
        let result = SystemSolver::solve_from_file_with_solver(mock_matrix_file, gmres_solver, preconditioner);

        // Validate results
        assert!(result.is_ok(), "Solver failed with error: {:?}", result.err());
        let solver_result = result.unwrap();
        assert!(solver_result.converged, "Solver did not converge");
        assert!(
            solver_result.residual_norm <= 1e-6,
            "Residual norm is too high: {}",
            solver_result.residual_norm
        );
    }
}
