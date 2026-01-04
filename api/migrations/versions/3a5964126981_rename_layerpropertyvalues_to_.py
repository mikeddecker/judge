"""rename LayerPropertyValues to LayerValues

Revision ID: 3a5964126981
Revises: 62083299fe22
Create Date: 2026-01-04 18:34:45.018652

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '3a5964126981'
down_revision = '62083299fe22'
branch_labels = None
depends_on = None

def upgrade():
    print('TRICKY MIGRATE')
    try:
            
        # -------------------------------------------------
        # 3. Update LayerValues.propertyId -> layerId
        # -------------------------------------------------
        op.drop_constraint(
            "LayerPropertyValues_ibfk_1",
            "LayerPropertyValues",
            type_="foreignkey",
        )

        # -------------------------------------------------
        # 1. Rename LayerProperties -> Layers
        # -------------------------------------------------
        op.rename_table("LayerProperties", "Layers")

        # -------------------------------------------------
        # 2. Rename LayerPropertyValues -> LayerValues
        # -------------------------------------------------
        op.rename_table("LayerPropertyValues", "LayerValues")


        op.alter_column(
            "LayerValues",
            "propertyId",
            new_column_name="layerId",
            existing_type=sa.Integer(),
            nullable=False,
        )

        op.create_foreign_key(
            "LayerValues_layerId_fk",
            "LayerValues",
            "Layers",
            ["layerId"],
            ["id"],
            ondelete="CASCADE",
        )

        # -------------------------------------------------
        # 4. Update LayerComposition.propertyId -> layerId
        # -------------------------------------------------
        op.drop_constraint(
            "LayerComposition_ibfk_1",
            "LayerComposition",
            type_="foreignkey",
        )

        op.alter_column(
            "LayerComposition",
            "propertyId",
            new_column_name="layerId",
            existing_type=sa.Integer(),
            nullable=False,
        )

        op.create_foreign_key(
            "LayerComposition_layerId_fk",
            "LayerComposition",
            "Layers",
            ["layerId"],
            ["id"],
            ondelete="CASCADE",
        )
        print('TRICKY MIGRATE SUCCESFULL')
    except Exception as e:
        print(e)

def downgrade():
    # -------------------------------------------------
    # Reverse LayerComposition
    # -------------------------------------------------
    op.drop_constraint(
        "LayerComposition_layerId_fk",
        "LayerComposition",
        type_="foreignkey",
    )

    op.alter_column(
        "LayerComposition",
        "layerId",
        new_column_name="propertyId",
        existing_type=sa.Integer(),
        nullable=False,
    )

    op.create_foreign_key(
        "LayerComposition_ibfk_1",
        "LayerComposition",
        "LayerProperties",
        ["propertyId"],
        ["id"],
        ondelete="CASCADE",
    )

    # -------------------------------------------------
    # Reverse LayerValues
    # -------------------------------------------------
    op.drop_constraint(
        "LayerValues_layerId_fk",
        "LayerValues",
        type_="foreignkey",
    )

    op.alter_column(
        "LayerValues",
        "layerId",
        new_column_name="propertyId",
        existing_type=sa.Integer(),
        nullable=False,
    )

    op.rename_table("LayerValues", "LayerPropertyValues")

    op.create_foreign_key(
        "LayerPropertyValues_ibfk_1",
        "LayerPropertyValues",
        "LayerProperties",
        ["propertyId"],
        ["id"],
        ondelete="CASCADE",
    )

    # -------------------------------------------------
    # Reverse Layers
    # -------------------------------------------------
    op.rename_table("Layers", "LayerProperties")
