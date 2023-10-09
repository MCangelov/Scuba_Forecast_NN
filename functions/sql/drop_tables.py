def drop_all_tables(engine):
    """
    Drop all tables in the 'public' schema of a PostgreSQL database.

    Parameters:
    -----------
    engine : sqlalchemy.engine.base.Engine
    """
    
    sql = """
    DO $$ 
    DECLARE 
        table_name text;
    BEGIN 
        FOR table_name IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public') 
        LOOP 
            EXECUTE 'DROP TABLE IF EXISTS public."' || table_name || '" CASCADE'; 
        END LOOP; 
    END $$;
    """

    with engine.connect() as connection:
        connection.execute(text(sql))
        connection.commit()