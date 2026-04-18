use anyhow::Result;
use clap::Parser;

pub mod blob;
pub mod branch;

#[derive(Parser)]
pub enum StoreCommand {
    /// Operations on branches stored in a remote object store.
    Branch {
        #[command(subcommand)]
        cmd: branch::Command,
    },
    /// Operations on blobs stored in a remote object store.
    Blob {
        #[command(subcommand)]
        cmd: blob::Command,
    },
}

pub fn run(cmd: StoreCommand) -> Result<()> {
    match cmd {
        StoreCommand::Branch { cmd } => branch::run(cmd),
        StoreCommand::Blob { cmd } => blob::run(cmd),
    }
}
