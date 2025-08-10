"""baseline

Revision ID: abb31821d630
Revises: 
Create Date: 2025-08-10 10:19:37.576508

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'abb31821d630'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Documents table additions
    with op.batch_alter_table('documents') as batch_op:
        batch_op.add_column(sa.Column('content', sa.Text()))
        batch_op.add_column(sa.Column('total_pages', sa.Integer()))
        batch_op.add_column(sa.Column('analysis_complete', sa.Boolean(), server_default=sa.text('0')))
        batch_op.add_column(sa.Column('uploaded_by', sa.String(length=100)))

    # Patterns table additions
    with op.batch_alter_table('patterns') as batch_op:
        batch_op.add_column(sa.Column('page_number', sa.Integer()))

    # Pages table additions
    with op.batch_alter_table('pages') as batch_op:
        batch_op.add_column(sa.Column('image_path', sa.String(length=500)))


def downgrade() -> None:
    # Revert Pages table additions
    with op.batch_alter_table('pages') as batch_op:
        batch_op.drop_column('image_path')

    # Revert Patterns table additions
    with op.batch_alter_table('patterns') as batch_op:
        batch_op.drop_column('page_number')

    # Revert Documents table additions
    with op.batch_alter_table('documents') as batch_op:
        batch_op.drop_column('uploaded_by')
        batch_op.drop_column('analysis_complete')
        batch_op.drop_column('total_pages')
        batch_op.drop_column('content')


