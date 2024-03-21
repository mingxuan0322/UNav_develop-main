CREATE TABLE cargo (
    flight_type_flight_type_id NUMBER NOT NULL,
    load                       VARCHAR2 
--  ERROR: VARCHAR2 size not specified 
     NOT NULL
);

ALTER TABLE cargo ADD CONSTRAINT cargo_pk PRIMARY KEY ( flight_type_flight_type_id );

CREATE TABLE charter (
    flight_type_flight_type_id NUMBER NOT NULL,
    owner                      VARCHAR2 
--  ERROR: VARCHAR2 size not specified 
     NOT NULL
);

ALTER TABLE charter ADD CONSTRAINT charter_pk PRIMARY KEY ( flight_type_flight_type_id );

CREATE TABLE domestic (
    flight_type_flight_type_id NUMBER NOT NULL,
    departure_city             VARCHAR2 
--  ERROR: VARCHAR2 size not specified 
     NOT NULL
);

ALTER TABLE domestic ADD CONSTRAINT domesticv1_pk PRIMARY KEY ( flight_type_flight_type_id );

CREATE TABLE international (
    flight_type_flight_type_id NUMBER NOT NULL,
    departure_country          VARCHAR2 
--  ERROR: VARCHAR2 size not specified 
     NOT NULL
);

ALTER TABLE international ADD CONSTRAINT internationalv1_pk PRIMARY KEY ( flight_type_flight_type_id );

CREATE TABLE xl_airline (
    airline_id                  VARCHAR2(2) NOT NULL,
    airline_name                VARCHAR2 
--  ERROR: VARCHAR2 size not specified 
    ,
    xl_flight_flight_id         VARCHAR2 
--  ERROR: VARCHAR2 size not specified 
     NOT NULL,
    xl_flight_departure_date    DATE NOT NULL,
    xl_flight_arrival_date      DATE NOT NULL,
    xl_flight_departure_airport VARCHAR2 
--  ERROR: VARCHAR2 size not specified 
     NOT NULL,
    xl_flight_arrival_airport   VARCHAR2 
--  ERROR: VARCHAR2 size not specified 
     NOT NULL,
    xl_airport_airport_code     VARCHAR2 
--  ERROR: VARCHAR2 size not specified 
     NOT NULL
);

ALTER TABLE xl_airline ADD CONSTRAINT airline_pk PRIMARY KEY ( airline_id );

CREATE TABLE xl_airport (
    airport_code VARCHAR2 
--  ERROR: VARCHAR2 size not specified 
     NOT NULL,
    full_address CLOB NOT NULL
);

ALTER TABLE xl_airport ADD CONSTRAINT xl_airport_pk PRIMARY KEY ( airport_code );

CREATE TABLE xl_flight (
    flight_id         VARCHAR2 
--  ERROR: VARCHAR2 size not specified 
     NOT NULL,
    departure_date    DATE NOT NULL,
    arrival_date      DATE NOT NULL,
    departure_airport VARCHAR2 
--  ERROR: VARCHAR2 size not specified 
     NOT NULL,
    arrival_airport   VARCHAR2 
--  ERROR: VARCHAR2 size not specified 
     NOT NULL,
    flight_type       VARCHAR2 
--  ERROR: VARCHAR2 size not specified 
     NOT NULL,
    airline_name      VARCHAR2 
--  ERROR: VARCHAR2 size not specified 
     NOT NULL
);

ALTER TABLE xl_flight
    ADD CONSTRAINT xl_flight_pk PRIMARY KEY ( flight_id,
                                              departure_date,
                                              arrival_date,
                                              departure_airport,
                                              arrival_airport );

CREATE TABLE xl_flight_type (
    flight_type_id              NUMBER NOT NULL,
    xl_flight_flight_id         VARCHAR2 
--  ERROR: VARCHAR2 size not specified 
     NOT NULL,
    xl_flight_departure_date    DATE NOT NULL,
    xl_flight_arrival_date      DATE NOT NULL,
    xl_flight_departure_airport VARCHAR2 
--  ERROR: VARCHAR2 size not specified 
     NOT NULL,
    xl_flight_arrival_airport   VARCHAR2 
--  ERROR: VARCHAR2 size not specified 
     NOT NULL,
    xl_airport_airport_code     VARCHAR2 
--  ERROR: VARCHAR2 size not specified 
     NOT NULL
);

ALTER TABLE xl_flight_type ADD CONSTRAINT flight_type_pk PRIMARY KEY ( flight_type_id );

ALTER TABLE xl_airline
    ADD CONSTRAINT airline_xl_airport_fk FOREIGN KEY ( xl_airport_airport_code )
        REFERENCES xl_airport ( airport_code );

ALTER TABLE xl_airline
    ADD CONSTRAINT airline_xl_flight_fk FOREIGN KEY ( xl_flight_flight_id,
                                                      xl_flight_departure_date,
                                                      xl_flight_arrival_date,
                                                      xl_flight_departure_airport,
                                                      xl_flight_arrival_airport )
        REFERENCES xl_flight ( flight_id,
                               departure_date,
                               arrival_date,
                               departure_airport,
                               arrival_airport );

ALTER TABLE cargo
    ADD CONSTRAINT cargo_flight_type_fk FOREIGN KEY ( flight_type_flight_type_id )
        REFERENCES xl_flight_type ( flight_type_id );

ALTER TABLE charter
    ADD CONSTRAINT charter_flight_type_fk FOREIGN KEY ( flight_type_flight_type_id )
        REFERENCES xl_flight_type ( flight_type_id );

ALTER TABLE domestic
    ADD CONSTRAINT domesticv1_flight_type_fk FOREIGN KEY ( flight_type_flight_type_id )
        REFERENCES xl_flight_type ( flight_type_id );

ALTER TABLE xl_flight_type
    ADD CONSTRAINT flight_type_xl_airport_fk FOREIGN KEY ( xl_airport_airport_code )
        REFERENCES xl_airport ( airport_code );

ALTER TABLE xl_flight_type
    ADD CONSTRAINT flight_type_xl_flight_fk FOREIGN KEY ( xl_flight_flight_id,
                                                          xl_flight_departure_date,
                                                          xl_flight_arrival_date,
                                                          xl_flight_departure_airport,
                                                          xl_flight_arrival_airport )
        REFERENCES xl_flight ( flight_id,
                               departure_date,
                               arrival_date,
                               departure_airport,
                               arrival_airport );

ALTER TABLE international
    ADD CONSTRAINT internationalv1_flight_type_fk FOREIGN KEY ( flight_type_flight_type_id )
        REFERENCES xl_flight_type ( flight_type_id );

--  ERROR: No Discriminator Column found in Arc FKArc_2 - constraint trigger for Arc cannot be generated 

--  ERROR: No Discriminator Column found in Arc FKArc_2 - constraint trigger for Arc cannot be generated 

--  ERROR: No Discriminator Column found in Arc FKArc_2 - constraint trigger for Arc cannot be generated 

--  ERROR: No Discriminator Column found in Arc FKArc_2 - constraint trigger for Arc cannot be generated

CREATE SEQUENCE xl_flight_type_flight_type_id START WITH 1 NOCACHE ORDER;

CREATE OR REPLACE TRIGGER xl_flight_type_flight_type_id BEFORE
    INSERT ON xl_flight_type
    FOR EACH ROW
    WHEN ( new.flight_type_id IS NULL )
BEGIN
    :new.flight_type_id := xl_flight_type_flight_type_id.nextval;
END;
/
