use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodeCausalType {
    VariableState,
    Assignment,
    FunctionCall,
    Conditional,
    Loop,
    ReturnValue,
    Exception,
    ImportDependency,
    TypeDefinition,
    Mutation,
    Assertion,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CodeAttributes {
    pub data_flow: Option<DataFlow>,
    pub control_flow: Option<ControlFlow>,
    pub scope_level: Option<u32>,
    pub side_effect: Option<bool>,
    pub pure: Option<bool>,
    pub variable: Option<String>,
    pub function: Option<String>,
    pub type_info: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataFlow {
    Read,
    Write,
    ReadWrite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlFlow {
    Branch,
    Merge,
    LoopEntry,
    LoopExit,
}
