"""Initial schema

Revision ID: 001_initial
Revises: 
Create Date: 2025-01-01 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'fusion_jobs',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('status', sa.String(20), nullable=True),
        sa.Column('progress', sa.Integer(), nullable=True),
        sa.Column('primary_language', sa.String(20), nullable=True),
        sa.Column('secondary_language', sa.String(20), nullable=True),
        sa.Column('target_language', sa.String(20), nullable=True),
        sa.Column('strategy', sa.String(20), nullable=True),
        sa.Column('primary_code', sa.Text(), nullable=True),
        sa.Column('secondary_code', sa.Text(), nullable=True),
        sa.Column('fused_code', sa.Text(), nullable=True),
        sa.Column('explanation', sa.Text(), nullable=True),
        sa.Column('test_cases', sa.Text(), nullable=True),
        sa.Column('agent_traces', sa.JSON(), nullable=True),
        sa.Column('warnings', sa.JSON(), nullable=True),
        sa.Column('cosine_similarity', sa.Float(), nullable=True),
        sa.Column('structural_overlap', sa.Float(), nullable=True),
        sa.Column('processing_time_ms', sa.Float(), nullable=True),
        sa.Column('tokens_used', sa.Integer(), nullable=True),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('complexity', sa.Float(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_fusion_jobs_status', 'fusion_jobs', ['status'])
    op.create_index('ix_fusion_jobs_created_at', 'fusion_jobs', ['created_at'])

    op.create_table(
        'code_index',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('code', sa.Text(), nullable=False),
        sa.Column('language', sa.String(20), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('namespace', sa.String(100), nullable=True),
        sa.Column('faiss_id', sa.Integer(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_code_index_language', 'code_index', ['language'])
    op.create_index('ix_code_index_namespace', 'code_index', ['namespace'])

    op.create_table(
        'system_stats',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('event', sa.String(50), nullable=True),
        sa.Column('value', sa.Float(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )


def downgrade() -> None:
    op.drop_table('system_stats')
    op.drop_table('code_index')
    op.drop_table('fusion_jobs')
